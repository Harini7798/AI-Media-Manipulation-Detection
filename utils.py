"""Shared utilities used across the deepfake detection pipeline and server."""

import os

# Allowed image extensions for upload validation
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

MAX_FILE_SIZE_MB = 50  # bumped to allow short videos
INPUT_SIZE = 128

# Video sampling: take at most this many frames from a video, evenly spaced
MAX_VIDEO_FRAMES = 30


def get_filename_only(file_path: str) -> str:
    """Return the filename without extension from a full path."""
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split(".")[0]
    return filename_only


def allowed_file(filename: str) -> bool:
    """Check if the filename has an allowed image OR video extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def is_video(filename: str) -> bool:
    """Check if a filename is a video based on extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_VIDEO_EXTENSIONS
