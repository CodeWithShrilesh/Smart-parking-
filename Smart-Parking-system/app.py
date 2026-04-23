import os
from uuid import uuid4

from flask import Flask, redirect, render_template, request, url_for
from detector import detect_parking, set_media_source

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "mp4", "avi"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    data = detect_parking()
    status = request.args.get("status", "")
    error = request.args.get("error", "")
    yolo_active = "YOLO active" in data["detection_mode"]
    return render_template(
        "index.html",
        total=data["total"],
        free=data["free"],

        occupied=data["occupied"],
        source=data["source"],
        source_name=data["source_name"],
        detection_mode=data["detection_mode"],
        is_video=data["source"] == "Video",
        frame_token=data["frame_token"],
        current_vehicles=data["current_vehicles"],
        history=data["history"],
        parking_summary=data["parking_summary"],
        status=status,
        error=error,
        yolo_status=data["detection_mode"],
        yolo_active=yolo_active,
    )


@app.route("/upload", methods=["POST"])
def upload_media():
    if "media_file" not in request.files:
        return redirect(url_for("home", error="No file selected."))

    file = request.files["media_file"]
    if file.filename == "":
        return redirect(url_for("home", error="No file selected."))

    if not allowed_file(file.filename):
        return redirect(url_for("home", error="Unsupported file type."))

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ext = file.filename.rsplit(".", 1)[1].lower()
    saved_name = f"uploaded_{uuid4().hex}.{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(saved_path)

    result = set_media_source(saved_path)
    if result["ok"]:
        message = (
            f"Uploaded {result['source_name']} | "
            f"Cars detected: {result['cars_detected']} | "
            f"Mode: {result['mode']}"
        )
        return redirect(url_for("home", status=message))

    return redirect(url_for("home", error=result["message"]))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
