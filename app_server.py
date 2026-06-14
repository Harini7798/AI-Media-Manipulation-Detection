import os
import pickle
import tempfile
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from utils import (
    ALLOWED_EXTENSIONS,
    INPUT_SIZE,
    MAX_FILE_SIZE_MB,
    MAX_VIDEO_FRAMES,
    allowed_file,
    is_video,
)

MODEL_PATH = os.path.join(".", "tmp_checkpoint", "best_model.h5")
SVM_PATH = os.path.join(".", "tmp_checkpoint", "svm_model.pkl")

app = Flask(__name__, static_folder="frontend", static_url_path="")

# ── Model loading ────────────────────────────────────────────────────────────
_backbone = None
_svm = None
_scaler = None
_init_error: Optional[str] = None

try:
    import tensorflow as tf

    from tensorflow.keras.applications import EfficientNetB0

    _backbone = EfficientNetB0(
        weights="imagenet",
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        include_top=False,
        pooling="max",
    )
    _backbone.predict(np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32), verbose=0)

    if os.path.isfile(SVM_PATH):
        with open(SVM_PATH, "rb") as f:
            data = pickle.load(f)
        _svm = data["svm"]
        _scaler = data["scaler"]
    else:
        _init_error = f"SVM model not found at {SVM_PATH}"
except Exception as exc:
    _init_error = f"Failed to load model: {exc}"


# ── Face detector (lazy loaded; falls back gracefully if MTCNN unavailable) ──
_face_detector = None
_face_detector_error: Optional[str] = None

try:
    from mtcnn import MTCNN

    _face_detector = MTCNN()
except Exception as exc:
    _face_detector_error = f"MTCNN unavailable, predictions will use full frames: {exc}"


def _crop_face(bgr_frame: np.ndarray, margin: float = 0.3, min_confidence: float = 0.9):
    """Detect the most confident face in a BGR frame and return a cropped BGR region.

    Returns (cropped_face, confidence) on success, or (None, 0.0) if no face was found
    or the detector is unavailable. The crop includes a configurable margin around
    the bounding box.
    """
    if _face_detector is None or bgr_frame is None or bgr_frame.size == 0:
        return None, 0.0

    # MTCNN expects RGB
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    try:
        results = _face_detector.detect_faces(rgb)
    except Exception:
        return None, 0.0

    if not results:
        return None, 0.0

    best = max(results, key=lambda r: r.get("confidence", 0.0))
    if best.get("confidence", 0.0) < min_confidence:
        return None, 0.0

    x, y, w, h = best["box"]
    mx, my = int(w * margin), int(h * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(rgb.shape[1], x + w + mx)
    y2 = min(rgb.shape[0], y + h + my)

    if x2 <= x1 or y2 <= y1:
        return None, 0.0

    crop = bgr_frame[y1:y2, x1:x2]
    if crop.shape[0] < 20 or crop.shape[1] < 20:
        return None, 0.0

    return crop, float(best["confidence"])


def _preprocess_image(file_bytes: bytes):
    """Decode uploaded image bytes, crop the face (if any), and return (tensor, used_face_crop)."""
    if not file_bytes:
        raise ValueError("Could not decode image — file is empty")
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — file may be corrupt or not an image")

    face_crop, _conf = _crop_face(img)
    if face_crop is not None:
        return _frame_to_tensor(face_crop), True
    return _frame_to_tensor(img), False


def _frame_to_tensor(bgr_frame: np.ndarray) -> np.ndarray:
    """Convert a single BGR frame (from cv2) to a model-ready tensor."""
    img = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def _predict_tensor(tensor: np.ndarray) -> float:
    """Run backbone + SVM on a preprocessed tensor and return pristine probability."""
    features = _backbone.predict(tensor, verbose=0).flatten().reshape(1, -1)
    features_scaled = _scaler.transform(features)
    return float(_svm.predict_proba(features_scaled)[0][1])


def _process_video(file_bytes: bytes):
    """Sample frames from a video, run predictions on each, return aggregate + per-frame results."""
    # Write bytes to a temp file (cv2 can't read from memory)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if total_frames < 1:
            raise ValueError("Video has no frames")

        # Pick evenly-spaced frame indices, capped at MAX_VIDEO_FRAMES
        n_samples = min(MAX_VIDEO_FRAMES, total_frames)
        if n_samples == 1:
            indices = [total_frames // 2]
        else:
            indices = np.linspace(0, total_frames - 1, n_samples, dtype=int).tolist()

        frame_scores = []  # pristine probability per sampled frame
        frame_timestamps = []
        frames_with_face = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            face_crop, _conf = _crop_face(frame)
            if face_crop is not None:
                tensor = _frame_to_tensor(face_crop)
                frames_with_face += 1
            else:
                # Skip frames with no detectable face — they would just confuse the model
                continue
            score = _predict_tensor(tensor)
            frame_scores.append(score)
            frame_timestamps.append(round(idx / fps, 3))

        cap.release()

        if not frame_scores:
            raise ValueError(
                "No frames with a detectable face were found in the video"
            )

        avg_pristine = float(np.mean(frame_scores))
        return {
            "pristine_prob": avg_pristine,
            "deepfake_prob": 1.0 - avg_pristine,
            "frame_scores": [round(s, 6) for s in frame_scores],
            "frame_timestamps": frame_timestamps,
            "frame_count": len(frame_scores),
            "frames_sampled": len(indices),
            "frames_with_face": frames_with_face,
            "duration_sec": round(total_frames / fps, 2),
            "fps": round(fps, 2),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/api/status")
def api_status():
    return jsonify(
        {
            "model_loaded": _svm is not None and _backbone is not None,
            "model_path": SVM_PATH,
            "init_error": _init_error,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
        }
    )


@app.post("/api/predict")
def api_predict():
    # ── validate presence ────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return jsonify({"ok": False, "error": "No file selected"}), 400

    # ── validate extension ───────────────────────────────────────────────
    if not allowed_file(file.filename):
        exts = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({"ok": False, "error": f"File type not allowed. Accepted: {exts}"}), 400

    # ── validate size ────────────────────────────────────────────────────
    file_bytes = file.read()
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return jsonify({"ok": False, "error": f"File exceeds {MAX_FILE_SIZE_MB} MB limit"}), 400
    if len(file_bytes) == 0:
        return jsonify({"ok": False, "error": "Uploaded file is empty"}), 400

    # ── model gate ───────────────────────────────────────────────────────
    if _svm is None or _backbone is None:
        return jsonify({"ok": False, "error": _init_error or "Model not loaded"}), 501

    # ── route to image or video processing ───────────────────────────────
    try:
        if is_video(file.filename):
            result = _process_video(file_bytes)
            label = "Pristine" if result["pristine_prob"] >= 0.5 else "Deepfake"
            return jsonify(
                {
                    "ok": True,
                    "label": label,
                    "pristine_prob": round(result["pristine_prob"], 6),
                    "deepfake_prob": round(result["deepfake_prob"], 6),
                    "is_video": True,
                    "frame_scores": result["frame_scores"],
                    "frame_timestamps": result["frame_timestamps"],
                    "frame_count": result["frame_count"],
                    "meta": {
                        "input_size": INPUT_SIZE,
                        "duration_sec": result["duration_sec"],
                        "fps": result["fps"],
                        "frames_sampled": result.get("frames_sampled"),
                        "frames_with_face": result.get("frames_with_face"),
                    },
                }
            )
        else:
            tensor, used_face_crop = _preprocess_image(file_bytes)
            pristine_prob = _predict_tensor(tensor)
            deepfake_prob = 1.0 - pristine_prob
            label = "Pristine" if pristine_prob >= 0.5 else "Deepfake"

            return jsonify(
                {
                    "ok": True,
                    "label": label,
                    "pristine_prob": round(pristine_prob, 6),
                    "deepfake_prob": round(deepfake_prob, 6),
                    "is_video": False,
                    "meta": {
                        "input_size": INPUT_SIZE,
                        "face_detected": used_face_crop,
                    },
                }
            )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    # Run: python app_server.py
    # Open: http://localhost:8000
    app.run(host="0.0.0.0", port=8000, debug=False)
