from __future__ import annotations
import json, os, shutil, time, uuid
from pathlib import Path
from fastapi import UploadFile, HTTPException
from .config import SESSIONS_DIR, RESULTS_DIR, SESSION_TTL_SECONDS, RESULT_TTL_SECONDS, MAX_UPLOAD_MB

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _safe_ext(name: str) -> str:
    ext = Path(name or "").suffix.lower()
    return ext if ext in ALLOWED_EXTENSIONS else ".bin"


async def save_upload(upload: UploadFile, target: Path, limit_mb: int | None = None) -> int:
    limit_mb = MAX_UPLOAD_MB if limit_mb is None else int(limit_mb)
    limit = limit_mb * 1024 * 1024
    total = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise HTTPException(413, f"Upload exceeds {limit_mb} MB limit")
                f.write(chunk)
    finally:
        await upload.close()
    return total


def new_session_dir() -> tuple[str, Path]:
    sid = uuid.uuid4().hex
    d = SESSIONS_DIR / sid
    d.mkdir(parents=True, exist_ok=False)
    return sid, d


def session_dir(session_id: str) -> Path:
    if not session_id or any(c not in "0123456789abcdef" for c in session_id.lower()) or len(session_id) != 32:
        raise HTTPException(400, "Invalid session id")
    d = SESSIONS_DIR / session_id
    if not d.is_dir():
        raise HTTPException(404, "Session not found or expired")
    os.utime(d, None)
    return d


def write_meta(d: Path, meta: dict) -> None:
    (d / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_meta(d: Path) -> dict:
    p = d / "session.json"
    if not p.exists():
        raise HTTPException(500, "Session metadata is missing")
    return json.loads(p.read_text(encoding="utf-8"))


def cleanup_old() -> None:
    now = time.time()
    for root, ttl in ((SESSIONS_DIR, SESSION_TTL_SECONDS), (RESULTS_DIR, RESULT_TTL_SECONDS)):
        if not root.exists():
            continue
        for p in root.iterdir():
            try:
                age = now - p.stat().st_mtime
                if age > ttl:
                    if p.is_dir(): shutil.rmtree(p, ignore_errors=True)
                    else: p.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
