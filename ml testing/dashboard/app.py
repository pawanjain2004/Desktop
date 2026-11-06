"""Flask-SocketIO server that receives vein detection frames from a Pi and
streams them to a browser dashboard."""

import os

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "vein-finder-dev-key")

TOKEN = os.environ.get("STREAM_TOKEN", "changeme")

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    max_http_buffer_size=2 * 1024 * 1024,
)

latest_frame = {"data": None, "mode": "enhanced", "fps": 0}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok", "connected": latest_frame["data"] is not None}


@socketio.on("connect")
def on_connect():
    if latest_frame["data"]:
        emit("frame", latest_frame)


@socketio.on("auth")
def on_auth(data):
    """Pi authenticates before streaming."""
    if data.get("token") == TOKEN:
        emit("auth_ok", {"status": "authenticated"})
    else:
        emit("auth_fail", {"status": "invalid token"})


@socketio.on("frame")
def on_frame(data):
    """Receive a frame from the Pi and broadcast to all browser viewers."""
    token = data.get("token", "")
    if token != TOKEN:
        return

    latest_frame["data"] = data.get("image")
    latest_frame["mode"] = data.get("mode", "unknown")
    latest_frame["fps"] = data.get("fps", 0)

    emit(
        "frame",
        {
            "data": latest_frame["data"],
            "mode": latest_frame["mode"],
            "fps": latest_frame["fps"],
        },
        broadcast=True,
        include_self=False,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
