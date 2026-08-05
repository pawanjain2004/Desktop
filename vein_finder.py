#!/usr/bin/env python3
"""
Reflection-mode vein finder for Raspberry Pi Camera Module 3 NoIR.

Pipeline: capture IR frame → grayscale → median filter → CLAHE → median filter
→ adaptive threshold → morphology → edge enhance → side-by-side display.

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
from picamera2 import Picamera2


def enhance_veins(
    gray: np.ndarray, clahe, median_ksize: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Return (enhanced grayscale, binary vein mask)."""
    denoised = cv2.medianBlur(gray, median_ksize)
    eq = clahe.apply(denoised)
    filtered = cv2.medianBlur(eq, 3)

    inv = cv2.bitwise_not(filtered)

    binary = cv2.adaptiveThreshold(
        inv,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        4,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    edges = cv2.Laplacian(filtered, cv2.CV_16S, ksize=3)
    edges = cv2.convertScaleAbs(edges)
    enhanced = cv2.addWeighted(eq, 0.75, edges, 0.45, 0)
    enhanced = cv2.bitwise_not(enhanced)

    return enhanced, binary


def overlay_veins(bgr: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """Tint detected vein pixels green on the original frame."""
    out = bgr.copy()
    mask = binary > 0
    out[mask] = (0, 220, 80)
    return cv2.addWeighted(bgr, 0.55, out, 0.45, 0)


def build_view(mode, bgr, gray, enhanced, binary, width):
    """Build the side-by-side frame (raw | processed)."""
    if mode == "enhanced":
        view = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    elif mode == "overlay":
        view = overlay_veins(bgr, binary)
    elif mode == "binary":
        view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    else:
        view = bgr

    raw_panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return np.hstack((raw_panel, view))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IR reflection vein finder")
    p.add_argument("--width", type=int, default=1280, help="Capture width")
    p.add_argument("--height", type=int, default=720, help="Capture height")
    p.add_argument("--fullscreen", action="store_true", help="Fullscreen window")
    p.add_argument(
        "--mode",
        choices=("enhanced", "overlay", "binary", "raw"),
        default="enhanced",
        help="Display mode (keys 1-4 also switch live)",
    )
    p.add_argument(
        "--median-size",
        type=int,
        default=5,
        choices=(3, 5, 7),
        help="Median filter kernel size for noise reduction",
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
    return p.parse_args()


def run_local(args, picam, clahe):
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
            enhanced, binary = enhance_veins(gray, clahe, args.median_size)
            combo = build_view(mode, bgr, gray, enhanced, binary, args.width)

            frames += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps = frames / (now - fps_t0)
                frames = 0
                fps_t0 = now

            label = f"{mode} | {fps:.0f} FPS | 1-4 modes | q quit"
            cv2.putText(combo, label, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 255, 40), 2, cv2.LINE_AA)
            cv2.putText(combo, "RAW IR", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(combo, "PROCESSED", (args.width + 12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

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


def run_stream(args, picam, clahe):
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
    try:
        client.connect(args.server, transports=["websocket"])
    except Exception as e:
        print(f"Cannot connect to {args.server}: {e}")
        print("Start the dashboard server first, or check the URL.")
        return

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
            enhanced, binary = enhance_veins(gray, clahe, args.median_size)
            combo = build_view(mode, bgr, gray, enhanced, binary, args.width)

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
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

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
            run_stream(args, picam, clahe)
        else:
            run_local(args, picam, clahe)
    finally:
        picam.stop()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
