#!/usr/bin/env python3
"""
Reflection-mode vein finder for Raspberry Pi Camera Module 3 NoIR.

Pipeline: capture IR frame → grayscale → median → CLAHE → hand-interior mask
→ Frangi vesselness (boundary suppressed) + U-Net segmentation → overlay.

Modes:
  Local display:  python3 vein_finder.py
  Stream to cloud: python3 vein_finder.py --stream --server https://your-app.onrender.com

Hardware expected:
  - Pi 4 + Camera Module 3 NoIR (imx708_noir)
  - 850 nm IR LED ring around camera (6–12 LEDs, ~45° angle)
  - Working distance ~10–20 cm, preferably low ambient light
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from picamera2 import Picamera2


class DoubleConv(nn.Module):
    """Two convolution, GroupNorm, and ReLU layers used by the checkpoint U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class VeinUNet(nn.Module):
    """Single-channel U-Net matching the layer shapes in best_model.pt."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(1, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        self.dec4 = DoubleConv(base_channels * 24, base_channels * 8)
        self.dec3 = DoubleConv(base_channels * 12, base_channels * 4)
        self.dec2 = DoubleConv(base_channels * 6, base_channels * 2)
        self.dec1 = DoubleConv(base_channels * 3, base_channels)
        self.output = nn.Conv2d(base_channels, 1, kernel_size=1)

    @staticmethod
    def _up_and_join(image: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            image, size=skip.shape[2:], mode="bilinear", align_corners=True
        )
        return torch.cat((image, skip), dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(image)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.dec4(self._up_and_join(bottleneck, enc4))
        dec3 = self.dec3(self._up_and_join(dec4, enc3))
        dec2 = self.dec2(self._up_and_join(dec3, enc2))
        dec1 = self.dec1(self._up_and_join(dec2, enc1))
        return self.output(dec1)


class MLVeinDetector:
    """Load the checkpoint and segment a grayscale camera frame."""

    def __init__(self, model_path: str, threshold: float | None = None) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"ML checkpoint not found: {model_path}. "
                "Pass --model PATH or use --disable-ml."
            )

        try:
            checkpoint = torch.load(
                model_path, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load ML checkpoint {model_path}: {exc}") from exc

        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise ValueError(
                "Unsupported checkpoint format: expected a dictionary with model_state."
            )

        self.patch_size = int(checkpoint.get("patch_size", 256))
        base_channels = int(checkpoint.get("base_channels", 16))
        self.threshold = float(
            checkpoint.get("threshold", 0.5) if threshold is None else threshold
        )
        if self.patch_size <= 0 or base_channels <= 0:
            raise ValueError("Checkpoint patch_size and base_channels must be positive.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("ML threshold must be between 0 and 1.")

        self.model = VeinUNet(base_channels)
        try:
            self.model.load_state_dict(checkpoint["model_state"], strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint weights do not match the reconstructed U-Net architecture: "
                f"{exc}"
            ) from exc
        self.model.eval()

    @torch.inference_mode()
    def predict(self, gray: np.ndarray) -> np.ndarray:
        """Return full-frame vein probabilities as uint8 values in the range 0–255."""
        resized = cv2.resize(
            gray,
            (self.patch_size, self.patch_size),
            interpolation=cv2.INTER_AREA,
        )
        image = torch.from_numpy(resized).to(torch.float32).div_(255.0)
        image = image.unsqueeze(0).unsqueeze(0)
        probabilities = torch.sigmoid(self.model(image))[0, 0].cpu().numpy()
        probabilities = cv2.resize(
            probabilities,
            (gray.shape[1], gray.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.clip(probabilities * 255.0, 0, 255).astype(np.uint8)


def frangi_vesselness(
    image: np.ndarray,
    sigmas: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0),
    beta: float = 0.5,
    c: float = 0.5,
) -> np.ndarray:
    """Multi-scale Frangi for dark tubular veins. Returns uint8 0–255."""
    img = 1.0 - (image.astype(np.float64) / 255.0)
    max_response = np.zeros_like(img)

    for sigma in sigmas:
        ksize = max(3, int(6 * sigma + 1)) | 1
        smoothed = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

        dxx = cv2.Sobel(smoothed, cv2.CV_64F, 2, 0, ksize=3)
        dyy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 2, ksize=3)
        dxy = cv2.Sobel(smoothed, cv2.CV_64F, 1, 1, ksize=3)

        scale = sigma * sigma
        dxx *= scale
        dyy *= scale
        dxy *= scale

        tmp = np.sqrt(((dxx - dyy) * 0.5) ** 2 + dxy * dxy)
        mu1 = 0.5 * (dxx + dyy) + tmp
        mu2 = 0.5 * (dxx + dyy) - tmp

        swap = np.abs(mu1) > np.abs(mu2)
        lambda1 = np.where(swap, mu2, mu1)
        lambda2 = np.where(swap, mu1, mu2)

        abs_l1 = np.abs(lambda1)
        abs_l2 = np.abs(lambda2)
        rb = abs_l1 / (abs_l2 + 1e-10)
        s2 = abs_l1 * abs_l1 + abs_l2 * abs_l2

        response = np.exp(-(rb * rb) / (2 * beta * beta))
        response *= 1.0 - np.exp(-s2 / (2 * c * c))
        response[lambda2 >= 0] = 0.0

        max_response = np.maximum(max_response, response)

    peak = float(max_response.max())
    if peak > 1e-8:
        return (max_response / peak * 255.0).astype(np.uint8)
    return np.zeros_like(image, dtype=np.uint8)


def hand_interior_mask(gray: np.ndarray, border_px: int = 28) -> np.ndarray:
    """Segment the arm/hand and exclude silhouette band from detection."""
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.any(mask > 0) and np.any(mask == 0):
        if gray[mask > 0].mean() < gray[mask == 0].mean():
            mask = cv2.bitwise_not(mask)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        mask = np.where(labels == best, 255, 0).astype(np.uint8)

    fill = mask.copy()
    h, w = fill.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(fill, ff_mask, (0, 0), 128)
    holes = np.where(fill == 0, 255, 0).astype(np.uint8)
    mask = cv2.bitwise_or(mask, holes)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return np.where(dist >= float(border_px), 255, 0).astype(np.uint8)


def suppress_boundary_responses(
    vesselness: np.ndarray, gray: np.ndarray, interior: np.ndarray
) -> np.ndarray:
    """Kill silhouette / strong-edge Frangi hits; keep faint interior ridges."""
    out = vesselness.copy()
    out[interior == 0] = 0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    if np.any(interior > 0):
        edge_thr = float(np.percentile(mag[interior > 0], 97))
        out[mag >= edge_thr] = 0

    local = cv2.GaussianBlur(gray, (25, 25), 0)
    darker = gray.astype(np.int16) <= (local.astype(np.int16) + 1)
    out[~darker] = 0
    return out


def enhance_veins(
    gray: np.ndarray,
    clahe,
    median_ksize: int = 5,
    frangi_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      enhanced: CLAHE with mild vein darkening
      binary: hard mask (debug)
      vesselness: continuous Frangi map for soft green overlay
    """
    denoised = cv2.medianBlur(gray, median_ksize)
    eq = clahe.apply(denoised)
    filtered = cv2.medianBlur(eq, 3)

    interior = hand_interior_mask(denoised, border_px=28)

    h, w = filtered.shape
    if frangi_scale < 1.0:
        small = cv2.resize(
            filtered,
            (max(1, int(w * frangi_scale)), max(1, int(h * frangi_scale))),
            interpolation=cv2.INTER_AREA,
        )
        vessel_small = frangi_vesselness(small)
        vesselness = cv2.resize(vessel_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        vesselness = frangi_vesselness(filtered)

    vesselness = suppress_boundary_responses(vesselness, filtered, interior)
    vesselness = cv2.bitwise_and(vesselness, interior)

    if np.any(vesselness > 0):
        peak = float(vesselness.max())
        vesselness = (vesselness.astype(np.float32) / peak * 255.0).astype(np.uint8)

    strength = vesselness.astype(np.float32) / 255.0
    boosted = filtered.astype(np.float32) * (1.0 - 0.30 * strength)
    p2, p98 = np.percentile(boosted, (2, 98))
    if p98 > p2:
        boosted = np.clip((boosted - p2) * (255.0 / (p98 - p2)), 0, 255)
    enhanced = boosted.astype(np.uint8)

    # Low threshold so faint central veins still light up green
    if np.any(vesselness > 0):
        thr = int(np.percentile(vesselness[vesselness > 0], 40))
        thr = max(12, min(thr, 40))
    else:
        thr = 18
    _, binary = cv2.threshold(vesselness, thr, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, interior)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k, iterations=1)

    return enhanced, binary, vesselness


def overlay_veins_soft(
    bgr: np.ndarray,
    vesselness: np.ndarray,
    thr: int = 15,
    ml_probability: np.ndarray | None = None,
    ml_threshold: float = 0.5,
) -> np.ndarray:
    """Show Frangi responses in green and model predictions in blue."""
    out = bgr.astype(np.float32)
    v = vesselness.astype(np.float32)
    alpha = np.clip((v - float(thr)) / max(1.0, 255.0 - float(thr)), 0.0, 1.0)
    alpha = np.power(alpha, 0.55)  # lift mid responses into visible green
    alpha3 = alpha[:, :, None]

    green = out.copy()
    green[:, :, 0] = out[:, :, 0] * 0.12
    green[:, :, 1] = np.clip(out[:, :, 1] * 0.20 + 235.0, 0, 255)
    green[:, :, 2] = out[:, :, 2] * 0.12

    result = out * (1.0 - alpha3) + green * alpha3
    if ml_probability is not None:
        probability = ml_probability.astype(np.float32) / 255.0
        ml_alpha = np.clip(
            (probability - ml_threshold) / max(1e-6, 1.0 - ml_threshold),
            0.0,
            1.0,
        )
        ml_alpha = (ml_alpha * 0.75)[:, :, None]
        blue = np.empty_like(result)
        blue[:, :, 0] = 255.0
        blue[:, :, 1] = out[:, :, 1] * 0.12
        blue[:, :, 2] = out[:, :, 2] * 0.12
        result = result * (1.0 - ml_alpha) + blue * ml_alpha

    return np.clip(result, 0, 255).astype(np.uint8)


def build_view(
    mode,
    bgr,
    gray,
    enhanced,
    binary,
    vesselness,
    ml_probability=None,
    ml_threshold=0.5,
):
    """Build a three-panel frame: raw IR | traditional processing | ML output."""
    if np.any(vesselness > 0):
        thr = max(10, int(np.percentile(vesselness[vesselness > 0], 35)))
    else:
        thr = 15
    thr = min(thr, 35)

    if mode == "enhanced":
        view = overlay_veins_soft(
            cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
            vesselness,
            thr=thr,
        )
    elif mode == "overlay":
        view = overlay_veins_soft(bgr, vesselness, thr=thr)
    elif mode == "binary":
        view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    elif mode == "raw":
        view = bgr
    else:
        view = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    raw_panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if ml_probability is None:
        ml_panel = np.zeros_like(bgr)
    else:
        ml_panel = overlay_veins_soft(
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            np.zeros_like(vesselness),
            ml_probability=ml_probability,
            ml_threshold=ml_threshold,
        )

    panels = (raw_panel, view, ml_panel)
    titles = (
        ("RAW IR", (220, 220, 220)),
        ("SOFTWARE LAYERS", (80, 255, 100)),
        ("ML MODEL" if ml_probability is not None else "ML DISABLED", (255, 160, 80)),
    )
    for index, (title, color) in enumerate(titles):
        cv2.putText(
            panels[index],
            title,
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return np.hstack(panels)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IR reflection vein finder")
    p.add_argument("--width", type=int, default=1280, help="Capture width")
    p.add_argument("--height", type=int, default=720, help="Capture height")
    p.add_argument("--fullscreen", action="store_true", help="Fullscreen window")
    p.add_argument(
        "--mode",
        choices=("enhanced", "overlay", "binary", "raw"),
        default="enhanced",
        help="Software-layers display mode (keys 1-4 also switch live)",
    )
    p.add_argument(
        "--median-size",
        type=int,
        default=5,
        choices=(3, 5, 7),
        help="Median filter kernel size for noise reduction",
    )
    p.add_argument(
        "--frangi-scale",
        type=float,
        default=0.5,
        help="Frangi processing scale 0.25–1.0 (lower = faster)",
    )
    p.add_argument(
        "--stream",
        action="store_true",
        help="Stream frames to remote dashboard instead of local display",
    )
    p.add_argument(
        "--server",
        type=str,
        default="http://localhost:5000",
        help="Dashboard server URL (used with --stream)",
    )
    p.add_argument(
        "--token",
        type=str,
        default=None,
        help="Auth token for dashboard (or set STREAM_TOKEN env var)",
    )
    p.add_argument(
        "--stream-fps",
        type=int,
        default=4,
        help="Target FPS when streaming (default 4)",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=60,
        help="JPEG quality for streamed frames (1-100, default 60)",
    )
    p.add_argument(
        "--model",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "best_model.pt"
        ),
        help="PyTorch checkpoint path (default: best_model.pt beside this script)",
    )
    p.add_argument(
        "--ml-threshold",
        type=float,
        default=None,
        help="Model probability threshold (default: value saved in checkpoint)",
    )
    p.add_argument(
        "--disable-ml",
        action="store_true",
        help="Disable the model and run only the original Frangi pipeline",
    )
    return p.parse_args()


def run_local(args, picam, clahe, detector):
    """Run with local OpenCV window display."""
    win = "Vein Finder"
    if args.fullscreen:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, args.width, args.height)

    mode = args.mode
    fps_t0 = time.time()
    frames = 0
    fps = 0.0

    print("Controls: 1=enhanced  2=overlay  3=binary  4=raw  q/ESC=quit")

    try:
        while True:
            rgb = picam.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            enhanced, binary, vesselness = enhance_veins(
                gray, clahe, args.median_size, args.frangi_scale
            )
            ml_probability = detector.predict(gray) if detector else None
            combo = build_view(
                mode,
                bgr,
                gray,
                enhanced,
                binary,
                vesselness,
                ml_probability,
                detector.threshold if detector else 0.5,
            )

            frames += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps = frames / (now - fps_t0)
                frames = 0
                fps_t0 = now

            pipeline = "Frangi + ML" if detector else "Frangi"
            label = f"{mode} | {fps:.0f} FPS | {pipeline} | 1-4 modes | q quit"
            cv2.putText(combo, label, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2, cv2.LINE_AA)
            cv2.imshow(win, combo)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("1"):
                mode = "enhanced"
            elif key == ord("2"):
                mode = "overlay"
            elif key == ord("3"):
                mode = "binary"
            elif key == ord("4"):
                mode = "raw"
    finally:
        cv2.destroyAllWindows()


def run_stream(args, picam, clahe, detector):
    """Stream processed frames to the remote dashboard via SocketIO."""
    import socketio as sio_lib

    token = args.token or os.environ.get("STREAM_TOKEN", "changeme")
    frame_interval = 1.0 / args.stream_fps

    client = sio_lib.Client(reconnection=True, reconnection_delay=2)

    @client.event
    def connect():
        print(f"Connected to dashboard: {args.server}")
        client.emit("auth", {"token": token})

    @client.event
    def auth_ok(data):
        print("Authenticated with dashboard")

    @client.event
    def auth_fail(data):
        print("ERROR: Authentication failed. Check your --token / STREAM_TOKEN.")

    @client.event
    def disconnect():
        print("Disconnected from dashboard, will retry...")

    print(f"Connecting to {args.server} ...")
    while True:
        try:
            client.connect(
                args.server,
                transports=["polling", "websocket"],
                wait_timeout=30,
            )
            break
        except Exception as e:
            print(f"Cannot connect to {args.server}: {e}")
            print("Retrying in 15 seconds...")
            time.sleep(15)

    mode = args.mode
    fps_t0 = time.time()
    frames = 0
    fps = 0.0
    last_send = 0.0

    print(f"Streaming at ~{args.stream_fps} FPS, JPEG quality {args.jpeg_quality}")
    print("Press Ctrl+C to stop.")

    try:
        while client.connected:
            rgb = picam.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            enhanced, binary, vesselness = enhance_veins(
                gray, clahe, args.median_size, args.frangi_scale
            )
            ml_probability = detector.predict(gray) if detector else None
            combo = build_view(
                mode,
                bgr,
                gray,
                enhanced,
                binary,
                vesselness,
                ml_probability,
                detector.threshold if detector else 0.5,
            )

            frames += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps = frames / (now - fps_t0)
                frames = 0
                fps_t0 = now

            if now - last_send < frame_interval:
                continue
            last_send = now

            _, buf = cv2.imencode(
                ".jpg", combo, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )
            b64 = base64.b64encode(buf).decode("ascii")

            try:
                client.emit("frame", {
                    "image": b64,
                    "mode": mode,
                    "fps": round(fps, 1),
                    "token": token,
                })
            except Exception:
                pass
    finally:
        client.disconnect()


def main() -> int:
    args = parse_args()
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    args.frangi_scale = max(0.25, min(1.0, args.frangi_scale))
    detector = (
        None
        if args.disable_ml
        else MLVeinDetector(args.model, args.ml_threshold)
    )

    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"},
        controls={"FrameRate": 30},
    )
    picam.configure(config)
    picam.start()
    time.sleep(0.4)

    try:
        if args.stream:
            run_stream(args, picam, clahe, detector)
        else:
            run_local(args, picam, clahe, detector)
    finally:
        picam.stop()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
