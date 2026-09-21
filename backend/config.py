from __future__ import annotations
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("PYUGEN_DATA_ROOT", "/data/180pyugen"))
SESSIONS_DIR = DATA_ROOT / "sessions"
RESULTS_DIR = DATA_ROOT / "results"
SESSION_TTL_SECONDS = int(os.environ.get("PYUGEN_SESSION_TTL", "7200"))
RESULT_TTL_SECONDS = int(os.environ.get("PYUGEN_RESULT_TTL", "7200"))
MAX_UPLOAD_MB = int(os.environ.get("PYUGEN_MAX_UPLOAD_MB", "120"))
MAX_VIDEO_UPLOAD_MB = int(os.environ.get("PYUGEN_MAX_VIDEO_UPLOAD_MB", "4096"))
MAX_OUTPUT_HEIGHT = int(os.environ.get("PYUGEN_MAX_OUTPUT_HEIGHT", "4096"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = os.environ.get("PYUGEN_QUEUE", "render")
INLINE_JOBS = os.environ.get("PYUGEN_INLINE_JOBS", "0") == "1"

for p in (SESSIONS_DIR, RESULTS_DIR):
    p.mkdir(parents=True, exist_ok=True)
