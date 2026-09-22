"""Background render-job entry points.

Jobs are intentionally stateless apart from a session directory and payload.
This makes the same functions usable by the inline development executor and by
external RQ workers.  Progress callbacks report coarse stages rather than
forcing browser clients to understand renderer internals.
"""
from __future__ import annotations
import os
import zipfile
from pathlib import Path
import cv2
from . import engine
from .config import RESULTS_DIR
from .profiles import resolve_profile
from .renderer import render_stereo_blocked
from .storage import session_dir, read_meta

try:
    from rq import get_current_job
except Exception:
    get_current_job = lambda: None


def _set_progress(value: float, stage: str, progress_hook=None):
    value = float(max(0.0, min(1.0, value)))
    job = get_current_job()
    if job is not None:
        job.meta["progress"] = value
        job.meta["stage"] = stage
        job.save_meta()
    if progress_hook is not None:
        progress_hook(value, stage)


def _write_output(path: Path, img, fmt: str, jpeg_quality: int):
    if fmt == "png":
        ok, enc = cv2.imencode(".png", img)
    else:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    if not ok:
        raise RuntimeError("OpenCV could not encode the output image")
    path.parent.mkdir(parents=True, exist_ok=True)
    enc.tofile(os.fspath(path))


def render_job(payload: dict, progress_hook=None) -> dict:
    sid = payload["session_id"]
    d = session_dir(sid)
    meta = read_meta(d)
    decoder = payload.get("decoder", "opencv")
    _set_progress(0.03, "Loading images", progress_hook)
    left = engine.read_image(str(d / meta["left_file"]), decoder)
    right = engine.read_image(str(d / meta["right_file"]), decoder)
    if left is None or right is None:
        raise RuntimeError("Could not decode one or both input images")

    _set_progress(0.08, "Preparing profiles", progress_hook)
    lp = resolve_profile(payload["left_profile"], "L", left.shape[1], left.shape[0])
    rp = resolve_profile(payload["right_profile"], "R", right.shape[1], right.shape[0])
    pts = payload["points"]
    n = int(payload["output_height"])

    def cb(frac):
        _set_progress(0.10 + 0.82*float(frac), "Rendering VR180", progress_hook)

    out = render_stereo_blocked(
        left, right, lp, rp,
        tuple(pts["A"][0]), tuple(pts["B"][0]),
        tuple(pts["A"][1]), tuple(pts["B"][1]),
        output_height=n,
        roll=float(payload.get("roll", 0)),
        pitch=float(payload.get("pitch", 0)),
        yaw=float(payload.get("yaw", 0)),
        block_rows=int(payload.get("block_rows", 32)),
        progress=cb,
    )
    out = engine.apply_right_eye_trim_sbs(
        out,
        float(payload.get("right_shift_x_deg",0.0)),
        float(payload.get("right_shift_y_deg",0.0)),
    )

    _set_progress(0.94, "Encoding output", progress_hook)
    job = get_current_job()
    job_id = job.id if job is not None else payload.get("job_id", "inline")
    result_dir = RESULTS_DIR / job_id
    fmt = payload.get("format", "jpeg").lower()
    ext = ".png" if fmt == "png" else ".jpg"
    requested = Path(payload.get("output_name") or ("LROut" + ext)).stem
    safe_stem = "".join(c for c in requested if c.isalnum() or c in "-_")[:80] or "LROut"
    result = result_dir / (safe_stem + ext)
    _write_output(result, out, fmt, int(payload.get("jpeg_quality", 95)))
    _set_progress(1.0, "Done", progress_hook)
    return {
        "kind": "image",
        "result_path": str(result),
        "filename": result.name,
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
        "format": fmt,
        "media_type": "image/png" if fmt == "png" else "image/jpeg",
    }


def render_video_job(payload: dict, progress_hook=None) -> dict:
    sid = payload["session_id"]
    d = session_dir(sid)
    meta = read_meta(d)
    if meta.get("kind") != "video":
        raise RuntimeError("Session is not a video session")

    left_path = d / meta["left_file"]
    right_path = d / meta["right_file"]
    li = meta["left_info"]
    ri = meta["right_info"]
    _set_progress(0.02, "Preparing video profiles", progress_hook)
    lp = resolve_profile(payload["left_profile"], "L", li["width"], li["height"])
    rp = resolve_profile(payload["right_profile"], "R", ri["width"], ri["height"])
    pts = payload["points"]

    job = get_current_job()
    job_id = job.id if job is not None else payload.get("job_id", "inline")
    result_dir = RESULTS_DIR / job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    requested = Path(payload.get("output_name") or "VROut.mp4")
    suffix = requested.suffix.lower()
    if suffix not in (".mp4",".avi",".mov",".m4v"):
        suffix = ".mp4"
    stem = "".join(c for c in requested.stem if c.isalnum() or c in "-_")[:80] or "VROut"
    out_path = result_dir / (stem + suffix)

    def cb(done,total,stage,frac):
        _set_progress(float(frac), stage, progress_hook)

    preview_path=result_dir/(stem+"_preview.mp4")
    r = engine.convert_stereo_video(
        str(left_path), str(right_path), str(out_path),
        lp, rp,
        tuple(pts["A"][0]), tuple(pts["B"][0]),
        tuple(pts["A"][1]), tuple(pts["B"][1]),
        int(payload["left_start"]), int(payload["left_end"]),
        int(payload["right_start"]), int(payload["right_end"]),
        output_width=int(payload.get("output_width",4096)),
        roll=float(payload.get("roll",0)), pitch=float(payload.get("pitch",0)), yaw=float(payload.get("yaw",0)),
        codec=str(payload.get("codec","auto")),
        fps_mode=str(payload.get("fps_mode","kino")),
        sampling=str(payload.get("sampling","fast")),
        length_policy=str(payload.get("length_policy","strict")),
        right_shift_x_deg=float(payload.get("right_shift_x_deg",0.0)),
        right_shift_y_deg=float(payload.get("right_shift_y_deg",0.0)),
        preview_path=str(preview_path),preview_width=1280,
        progress=cb,
    )
    # The H.264 proxy is written concurrently by the core renderer, avoiding a
    # costly second decode of the full-resolution HEVC result.
    preview_path=Path(r.get("preview_path") or preview_path)
    preview_error=r.get("preview_error")

    _set_progress(1.0, "Done", progress_hook)
    return {
        "kind":"video",
        "result_path":str(out_path),
        "filename":out_path.name,
        "result_bytes":int(out_path.stat().st_size) if out_path.exists() else 0,
        "preview_path":str(preview_path) if preview_path.exists() else None,
        "preview_bytes":int(preview_path.stat().st_size) if preview_path.exists() else 0,
        "preview_error":preview_error,
        "width":int(r["width"]),
        "height":int(r["height"]),
        "fps":float(r["fps"]),
        "frames":int(r.get("frames",0)),
        "left_frames":int(r.get("left_frames",0)),
        "right_frames":int(r.get("right_frames",0)),
        "length_policy":str(r.get("length_policy","strict")),
        "source_fps":float(r.get("source_fps",r["fps"])),
        "fps_mode":r.get("fps_mode","source"),
        "codec":r["codec"],
        "audio":False,
        "sampling":r["sampling"],
        "right_shift_x_deg":float(r.get("right_shift_x_deg",0.0)),
        "right_shift_y_deg":float(r.get("right_shift_y_deg",0.0)),
        "media_type":"video/mp4" if suffix in (".mp4",".m4v",".mov") else "video/x-msvideo",
    }


def clip_video_job(payload: dict, progress_hook=None) -> dict:
    sid = payload["session_id"]
    d = session_dir(sid)
    meta = read_meta(d)
    if meta.get("kind") != "video":
        raise RuntimeError("Session is not a video session")

    job = get_current_job()
    job_id = job.id if job is not None else payload.get("job_id", "inline")
    result_dir = RESULTS_DIR / job_id
    clip_dir = result_dir / "cut_img"
    result_dir.mkdir(parents=True, exist_ok=True)

    ls, le, rs, re = (int(payload[k]) for k in ("left_start", "left_end", "right_start", "right_end"))
    total = max(0, le-ls+1) + max(0, re-rs+1)
    if total <= 0:
        raise ValueError("Clip ranges are empty")

    def cb(done, total_count, stage):
        _set_progress(0.03 + 0.87*(float(done)/max(1,total_count)), stage, progress_hook)

    _set_progress(0.02, "Preparing JPEG clipping", progress_hook)
    files = engine.clip_video_jpegs(
        str(d / meta["left_file"]), str(d / meta["right_file"]), str(clip_dir),
        ls, le, rs, re, int(payload.get("jpeg_quality", 95)), cb
    )
    _set_progress(0.92, "Packing JPEG ZIP", progress_hook)
    zip_path = result_dir / "cut_img.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            fp = Path(f)
            z.write(fp, arcname=f"cut_img/{fp.name}")
    _set_progress(1.0, "Done", progress_hook)
    return {
        "kind": "video_clips",
        "result_path": str(zip_path),
        "filename": zip_path.name,
        "frames": len(files),
        "media_type": "application/zip",
    }
