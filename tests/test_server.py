"""Tests for the Flask prediction server and image preprocessing."""

import io
import struct
import zlib
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest


def _make_png_bytes(width=64, height=64) -> bytes:
    """Create a minimal valid PNG image in memory."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _make_jpeg_bytes(width=64, height=64) -> bytes:
    """Create a minimal valid JPEG image in memory."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


# ── Preprocessing tests (no TF needed) ──────────────────────────────────────

class TestPreprocessImage:
    """Test the _preprocess_image helper directly.

    _preprocess_image returns (tensor, used_face_crop). For random noise images
    MTCNN won't find a face so used_face_crop will be False — that's fine, we
    only validate the tensor shape and dtype here.
    """

    def test_valid_png_returns_correct_shape(self):
        from app_server import _preprocess_image
        tensor, _ = _preprocess_image(_make_png_bytes(200, 150))
        assert tensor.shape == (1, 128, 128, 3)

    def test_valid_jpeg_returns_correct_shape(self):
        from app_server import _preprocess_image
        tensor, _ = _preprocess_image(_make_jpeg_bytes(300, 300))
        assert tensor.shape == (1, 128, 128, 3)

    def test_output_normalised_to_0_1(self):
        from app_server import _preprocess_image
        tensor, _ = _preprocess_image(_make_png_bytes())
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_output_dtype_float32(self):
        from app_server import _preprocess_image
        tensor, _ = _preprocess_image(_make_png_bytes())
        assert tensor.dtype == np.float32

    def test_returns_tuple_with_face_flag(self):
        from app_server import _preprocess_image
        result = _preprocess_image(_make_png_bytes())
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[1], bool)

    def test_corrupt_bytes_raises(self):
        from app_server import _preprocess_image
        with pytest.raises(ValueError, match="Could not decode"):
            _preprocess_image(b"not-an-image")

    def test_empty_bytes_raises(self):
        from app_server import _preprocess_image
        with pytest.raises(ValueError, match="Could not decode"):
            _preprocess_image(b"")

    def test_small_image_upscaled(self):
        from app_server import _preprocess_image
        tensor, _ = _preprocess_image(_make_png_bytes(10, 10))
        assert tensor.shape == (1, 128, 128, 3)

    def test_large_image_downscaled(self):
        from app_server import _preprocess_image
        tensor, _ = _preprocess_image(_make_png_bytes(1024, 768))
        assert tensor.shape == (1, 128, 128, 3)


class TestCropFace:
    """Test the _crop_face helper. Random noise won't contain a face, so we
    validate the no-face fallback behavior."""

    def test_no_face_returns_none(self):
        from app_server import _crop_face
        # Random noise — MTCNN should find no face
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        crop, conf = _crop_face(frame)
        assert crop is None
        assert conf == 0.0

    def test_empty_frame_returns_none(self):
        from app_server import _crop_face
        crop, conf = _crop_face(np.array([], dtype=np.uint8))
        assert crop is None
        assert conf == 0.0

    def test_none_frame_returns_none(self):
        from app_server import _crop_face
        crop, conf = _crop_face(None)
        assert crop is None
        assert conf == 0.0


# ── Flask endpoint tests (model mocked) ─────────────────────────────────────

@pytest.fixture()
def client():
    """Create a Flask test client with the model mocked out."""
    import app_server
    # Mock backbone that returns a feature vector
    mock_backbone = MagicMock()
    mock_backbone.predict.return_value = np.zeros((1, 1280), dtype=np.float32)

    # Mock SVM that returns [deepfake_prob, pristine_prob]
    mock_svm = MagicMock()
    mock_svm.predict_proba.return_value = np.array([[0.15, 0.85]])

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 1280))

    orig = (app_server._backbone, app_server._svm, app_server._scaler, app_server._init_error)
    app_server._backbone = mock_backbone
    app_server._svm = mock_svm
    app_server._scaler = mock_scaler
    app_server._init_error = None
    app_server.app.config["TESTING"] = True

    with app_server.app.test_client() as c:
        yield c

    app_server._backbone, app_server._svm, app_server._scaler, app_server._init_error = orig


@pytest.fixture()
def client_no_model():
    """Create a Flask test client with no model loaded."""
    import app_server

    orig = (app_server._backbone, app_server._svm, app_server._scaler, app_server._init_error)
    app_server._backbone = None
    app_server._svm = None
    app_server._scaler = None
    app_server._init_error = "Model not loaded for testing"
    app_server.app.config["TESTING"] = True

    with app_server.app.test_client() as c:
        yield c

    app_server._backbone, app_server._svm, app_server._scaler, app_server._init_error = orig


class TestStatusEndpoint:
    def test_status_returns_json(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "model_loaded" in data

    def test_status_model_loaded(self, client):
        data = client.get("/api/status").get_json()
        assert data["model_loaded"] is True

    def test_status_model_not_loaded(self, client_no_model):
        data = client_no_model.get("/api/status").get_json()
        assert data["model_loaded"] is False
        assert data["init_error"] is not None


class TestPredictEndpoint:
    def test_predict_valid_png(self, client):
        data = {"file": (io.BytesIO(_make_png_bytes()), "face.png")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True
        assert body["label"] == "Pristine"
        assert 0 <= body["pristine_prob"] <= 1
        assert 0 <= body["deepfake_prob"] <= 1

    def test_predict_valid_jpeg(self, client):
        data = {"file": (io.BytesIO(_make_jpeg_bytes()), "face.jpg")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_predict_no_file(self, client):
        resp = client.post("/api/predict", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert resp.get_json()["ok"] is False

    def test_predict_empty_filename(self, client):
        data = {"file": (io.BytesIO(_make_png_bytes()), "")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_predict_disallowed_extension(self, client):
        data = {"file": (io.BytesIO(b"fake"), "hack.exe")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "not allowed" in resp.get_json()["error"]

    def test_predict_empty_file(self, client):
        data = {"file": (io.BytesIO(b""), "empty.png")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "empty" in resp.get_json()["error"].lower()

    def test_predict_model_not_loaded(self, client_no_model):
        data = {"file": (io.BytesIO(_make_png_bytes()), "face.png")}
        resp = client_no_model.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 501
        assert resp.get_json()["ok"] is False

    def test_predict_deepfake_label(self, client):
        """When model returns < 0.5, label should be Deepfake."""
        import app_server
        app_server._svm.predict_proba.return_value = np.array([[0.85, 0.15]])
        data = {"file": (io.BytesIO(_make_png_bytes()), "face.png")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        body = resp.get_json()
        assert body["label"] == "Deepfake"
        assert body["deepfake_prob"] > body["pristine_prob"]


class TestPredictVideoEndpoint:
    """Tests for video upload handling. _process_video is mocked because it
    needs a real video file on disk; we only verify the routing/response shape."""

    def test_video_routes_through_process_video(self, client):
        import app_server

        fake_result = {
            "pristine_prob": 0.92,
            "deepfake_prob": 0.08,
            "frame_scores": [0.91, 0.93, 0.92],
            "frame_timestamps": [0.0, 0.5, 1.0],
            "frame_count": 3,
            "duration_sec": 1.5,
            "fps": 30.0,
        }
        with patch.object(app_server, "_process_video", return_value=fake_result):
            data = {"file": (io.BytesIO(b"fake-mp4-data"), "clip.mp4")}
            resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["ok"] is True
            assert body["is_video"] is True
            assert body["label"] == "Pristine"
            assert body["frame_count"] == 3
            assert body["frame_scores"] == [0.91, 0.93, 0.92]
            assert body["frame_timestamps"] == [0.0, 0.5, 1.0]
            assert body["meta"]["duration_sec"] == 1.5
            assert body["meta"]["fps"] == 30.0

    def test_video_deepfake_label(self, client):
        import app_server

        fake_result = {
            "pristine_prob": 0.12,
            "deepfake_prob": 0.88,
            "frame_scores": [0.10, 0.15, 0.11],
            "frame_timestamps": [0.0, 0.5, 1.0],
            "frame_count": 3,
            "duration_sec": 1.5,
            "fps": 30.0,
        }
        with patch.object(app_server, "_process_video", return_value=fake_result):
            data = {"file": (io.BytesIO(b"fake-mp4-data"), "clip.mp4")}
            resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
            body = resp.get_json()
            assert body["label"] == "Deepfake"
            assert body["is_video"] is True

    def test_video_invalid_extension_rejected(self, client):
        data = {"file": (io.BytesIO(b"fake"), "clip.xyz")}
        resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "not allowed" in resp.get_json()["error"]

    def test_video_decode_failure_returns_400(self, client):
        import app_server
        with patch.object(app_server, "_process_video", side_effect=ValueError("Could not open video file")):
            data = {"file": (io.BytesIO(b"garbage"), "clip.mp4")}
            resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
            assert resp.status_code == 400
            assert resp.get_json()["ok"] is False
            assert "video" in resp.get_json()["error"].lower()


class TestIndexPage:
    def test_index_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"AI edited Media detector" in resp.data
