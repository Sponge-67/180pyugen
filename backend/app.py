"""FastAPI transport layer for 180pyugen Web.

The API intentionally does not duplicate camera mathematics.  Browser requests
are converted into temporary sessions and render-job payloads; authoritative
projection, stereo alignment, matching, and video remapping live in
backend.engine (the same Python core as the desktop application).

Video session lifecycle:
  upload pair -> probe streams -> fetch/display reference frames -> match A/B
  -> queue conversion -> poll progress -> download result -> expire session.

Development mode executes jobs in a single background thread.  Production mode
can enqueue the same payloads in Redis/RQ workers so web request threads never
perform long 4K/8K conversions themselves.
"""
from __future__ import annotations
import json, os, shutil, threading, uuid
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import engine
from .config import MAX_OUTPUT_HEIGHT, MAX_VIDEO_UPLOAD_MB, REDIS_URL, QUEUE_NAME, INLINE_JOBS, RESULTS_DIR
from .profiles import profile_names, profile_info, resolve_profile, GOPRO_L, GOPRO_R
from .schemas import MatchRequest, RenderRequest, VideoMatchRequest, VideoRenderRequest, VideoClipRequest, AlignmentPreviewRequest, VideoAlignmentPreviewRequest
from .storage import new_session_dir, save_upload, write_meta, read_meta, session_dir, cleanup_old
from .jobs import render_job, render_video_job, clip_video_job

app = FastAPI(title="180pyugen Web API", version="2.10")

_queue = None
_redis = None
if not INLINE_JOBS:
    try:
        import redis
        from rq import Queue
        _redis = redis.from_url(REDIS_URL)
        _queue = Queue(QUEUE_NAME, connection=_redis, default_timeout=60*60)
    except Exception:
        _queue = None

_inline_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="180pyugen")
_inline_jobs: dict[str, dict] = {}
_inline_lock = threading.Lock()


def _inline_submit(payload: dict, func=render_job) -> str:
    jid = uuid.uuid4().hex
    with _inline_lock:
        _inline_jobs[jid] = {"status":"queued", "progress":0.0, "stage":"Queued"}
    def hook(progress, stage):
        with _inline_lock:
            if jid in _inline_jobs:
                _inline_jobs[jid].update(progress=float(progress), stage=str(stage), status="started")
    def run():
        try:
            with _inline_lock:
                _inline_jobs[jid].update(status="started", stage="Preparing")
            p = dict(payload); p["job_id"] = jid
            result = func(p, progress_hook=hook)
            with _inline_lock:
                _inline_jobs[jid].update(status="finished", progress=1.0, stage="Done", result=result)
        except Exception as e:
            with _inline_lock:
                _inline_jobs[jid].update(status="failed", stage="Failed", error=str(e))
    _inline_pool.submit(run)
    return jid


@app.get("/api/health")
def health():
    cleanup_old()
    queue_ok = False
    if _redis is not None:
        try: queue_ok = bool(_redis.ping())
        except Exception: queue_ok = False
    return {"ok": True, "queue": "inline" if INLINE_JOBS else ("redis" if queue_ok else "unavailable")}


@app.get("/api/profiles")
def profiles():
    return {"profiles": profile_names(), "defaults": {"left": GOPRO_L, "right": GOPRO_R}, "max_output_height": MAX_OUTPUT_HEIGHT}


@app.get("/api/profile-info")
def get_profile_info(name: str, side: str, width: int, height: int):
    try:
        return profile_info(name, side, width, height)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/session")
async def create_session(left: UploadFile = File(...), right: UploadFile = File(...)):
    cleanup_old()
    sid, d = new_session_dir()
    try:
        left_name = "left" + Path(left.filename or "left.jpg").suffix.lower()
        right_name = "right" + Path(right.filename or "right.jpg").suffix.lower()
        await save_upload(left, d / left_name)
        await save_upload(right, d / right_name)
        L = engine.read_image(str(d / left_name), "opencv")
        R = engine.read_image(str(d / right_name), "opencv")
        if L is None or R is None:
            raise HTTPException(400, "Could not decode one or both images")
        meta = {
            "kind": "image",
            "left_file": left_name,
            "right_file": right_name,
            "left_size": [int(L.shape[1]), int(L.shape[0])],
            "right_size": [int(R.shape[1]), int(R.shape[0])],
            "left_original_name": left.filename,
            "right_original_name": right.filename,
        }
        write_meta(d, meta)
        return {"session_id": sid, **meta}
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


@app.post("/api/video/session")
async def create_video_session(left: UploadFile = File(...), right: UploadFile = File(...)):
    cleanup_old()
    sid, d = new_session_dir()
    try:
        left_ext = Path(left.filename or "left.mp4").suffix.lower() or ".mp4"
        right_ext = Path(right.filename or "right.mp4").suffix.lower() or ".mp4"
        left_name = "left_video" + left_ext
        right_name = "right_video" + right_ext
        await save_upload(left, d / left_name, limit_mb=MAX_VIDEO_UPLOAD_MB)
        await save_upload(right, d / right_name, limit_mb=MAX_VIDEO_UPLOAD_MB)
        li = engine.video_probe(str(d / left_name))
        ri = engine.video_probe(str(d / right_name))
        meta = {
            "kind":"video",
            "left_file":left_name, "right_file":right_name,
            "left_info":li, "right_info":ri,
            "left_original_name":left.filename, "right_original_name":right.filename,
        }
        write_meta(d, meta)
        return {"session_id":sid, **meta}
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


@app.get("/api/video/frame/{session_id}/{side}")
def video_frame(session_id: str, side: str, frame: int = 0):
    d = session_dir(session_id)
    meta = read_meta(d)
    if meta.get("kind") != "video":
        raise HTTPException(400, "Not a video session")
    side = side.upper()
    if side not in ("L","R"):
        raise HTTPException(400, "side must be L or R")
    path = d / meta["left_file" if side=="L" else "right_file"]
    try:
        img = engine.read_video_frame(str(path), int(frame))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok: raise RuntimeError("Could not encode preview frame")
        return Response(content=enc.tobytes(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/video/match")
def video_match(req: VideoMatchRequest):
    d = session_dir(req.session_id)
    meta = read_meta(d)
    if meta.get("kind") != "video":
        raise HTTPException(400, "Not a video session")
    try:
        L = engine.read_video_frame(str(d/meta["left_file"]), req.left_frame)
        R = engine.read_video_frame(str(d/meta["right_file"]), req.right_frame)
        rx,ry,score = engine.template_match_point_kino(L,R,req.x,req.y,20)
        return {"x":rx,"y":ry,"sqdiff":score}
    except Exception as e:
        raise HTTPException(400, str(e))




def _projected_alignment_response(L, R, left_profile, right_profile, points,
                                  roll, pitch, yaw, shift_x_deg, shift_y_deg,
                                  preview_height):
    """Render the same final-eye geometry used by the real output.

    The browser receives a tiny SBS PNG (left eye | right eye).  It can then
    interactively shift the projected right eye without rerunning the expensive
    fisheye/3-D transform.  The suggested phase-correlation shift is returned
    in headers and is also measured in these final-eye pixels.
    """
    hL,wL=L.shape[:2];hR,wR=R.shape[:2];n=int(preview_height)
    lp=resolve_profile(left_profile,"L",wL,hL);rp=resolve_profile(right_profile,"R",wR,hR)
    A=points.get("A");B=points.get("B")
    if not A or not B:raise ValueError("A/B coordinates are required")
    out=engine.render_stereo_exe_geometry(L,R,lp,rp,A[0],B[0],A[1],B[1],n,roll,pitch,yaw)
    out=engine.apply_right_eye_trim_sbs(out,shift_x_deg,shift_y_deg)
    left=out[:,:n];right=out[:,n:]
    gl=cv2.Laplacian(cv2.cvtColor(left,cv2.COLOR_BGR2GRAY).astype('float32'),cv2.CV_32F)
    gr=cv2.Laplacian(cv2.cvtColor(right,cv2.COLOR_BGR2GRAY).astype('float32'),cv2.CV_32F)
    win=cv2.createHanningWindow((n,n),cv2.CV_32F)
    (dx,dy),response=cv2.phaseCorrelate(gr,gl,win)
    dx=max(-n*.25,min(n*.25,float(dx)));dy=max(-n*.25,min(n*.25,float(dy)))
    ok,enc=cv2.imencode('.png',out)
    if not ok:raise RuntimeError("Could not encode alignment preview")
    return Response(content=enc.tobytes(),media_type='image/png',headers={
        'X-Align-DX':f'{dx:.9g}','X-Align-DY':f'{dy:.9g}','X-Align-Response':f'{float(response):.9g}',
        'X-Preview-Eye':str(n),'Cache-Control':'no-store, max-age=0'})

@app.post('/api/alignment-preview')
def alignment_preview(req: AlignmentPreviewRequest):
    d=session_dir(req.session_id);meta=read_meta(d)
    if meta.get('kind')!='image':raise HTTPException(400,'Not an image session')
    try:
        L=engine.read_image(str(d/meta['left_file']),req.decoder);R=engine.read_image(str(d/meta['right_file']),req.decoder)
        if L is None or R is None:raise ValueError('Could not decode images')
        return _projected_alignment_response(L,R,req.left_profile,req.right_profile,req.points,req.roll,req.pitch,req.yaw,req.right_shift_x_deg,req.right_shift_y_deg,req.preview_height)
    except Exception as e:raise HTTPException(400,str(e))

@app.post('/api/video/alignment-preview')
def video_alignment_preview(req: VideoAlignmentPreviewRequest):
    d=session_dir(req.session_id);meta=read_meta(d)
    if meta.get('kind')!='video':raise HTTPException(400,'Not a video session')
    try:
        L=engine.read_video_frame(str(d/meta['left_file']),req.left_frame);R=engine.read_video_frame(str(d/meta['right_file']),req.right_frame)
        return _projected_alignment_response(L,R,req.left_profile,req.right_profile,req.points,req.roll,req.pitch,req.yaw,req.right_shift_x_deg,req.right_shift_y_deg,req.preview_height)
    except Exception as e:raise HTTPException(400,str(e))


@app.post("/api/video/render")
def video_render(req: VideoRenderRequest):
    if req.output_width // 2 > MAX_OUTPUT_HEIGHT:
        raise HTTPException(400, f"Output width exceeds server limit ({MAX_OUTPUT_HEIGHT*2})")
    d = session_dir(req.session_id)
    meta = read_meta(d)
    if meta.get("kind") != "video":
        raise HTTPException(400, "Not a video session")
    payload = req.model_dump()
    if INLINE_JOBS:
        jid = _inline_submit(payload, render_video_job)
        return {"job_id":jid,"status":"queued"}
    if _queue is None:
        raise HTTPException(503, "Render queue is unavailable")
    job = _queue.enqueue("backend.jobs.render_video_job", payload,
                         job_timeout=24*60*60, result_ttl=2*60*60, failure_ttl=2*60*60)
    return {"job_id":job.id,"status":"queued"}


@app.post("/api/video/clip")
def video_clip(req: VideoClipRequest):
    d = session_dir(req.session_id)
    meta = read_meta(d)
    if meta.get("kind") != "video":
        raise HTTPException(400, "Not a video session")
    if req.left_end < req.left_start or req.right_end < req.right_start:
        raise HTTPException(400, "End frame must be >= start frame")
    payload=req.model_dump()
    if INLINE_JOBS:
        jid=_inline_submit(payload, clip_video_job)
        return {"job_id":jid,"status":"queued"}
    if _queue is None:
        raise HTTPException(503, "Render queue is unavailable")
    job=_queue.enqueue("backend.jobs.clip_video_job",payload,job_timeout=60*60,result_ttl=2*60*60,failure_ttl=2*60*60)
    return {"job_id":job.id,"status":"queued"}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    d = session_dir(session_id)
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


@app.post("/api/match")
def match(req: MatchRequest):
    d = session_dir(req.session_id)
    meta = read_meta(d)
    L = engine.read_image(str(d / meta["left_file"]), req.decoder)
    R = engine.read_image(str(d / meta["right_file"]), req.decoder)
    if L is None or R is None:
        raise HTTPException(400, "Could not decode images")
    try:
        rx, ry, score = engine.template_match_point(L, R, req.x, req.y, 20)
        return {"x": rx, "y": ry, "sqdiff": score}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/render")
def render(req: RenderRequest):
    if req.output_height > MAX_OUTPUT_HEIGHT:
        raise HTTPException(400, f"Output height exceeds server limit ({MAX_OUTPUT_HEIGHT})")
    session_dir(req.session_id)  # validation/touch
    payload = req.model_dump()
    if INLINE_JOBS:
        jid = _inline_submit(payload)
        return {"job_id": jid, "status": "queued"}
    if _queue is None:
        raise HTTPException(503, "Render queue is unavailable")
    job = _queue.enqueue("backend.jobs.render_job", payload, job_timeout=60*60, result_ttl=2*60*60, failure_ttl=2*60*60)
    return {"job_id": job.id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if INLINE_JOBS:
        with _inline_lock:
            j = _inline_jobs.get(job_id)
            if not j: raise HTTPException(404, "Job not found")
            return {"job_id": job_id, **j, "result_url": f"/api/jobs/{job_id}/result" if j.get("status") == "finished" else None}
    if _redis is None:
        raise HTTPException(503, "Render queue is unavailable")
    from rq.job import Job
    try: job = Job.fetch(job_id, connection=_redis)
    except Exception: raise HTTPException(404, "Job not found")
    status = job.get_status(refresh=True)
    body = {
        "job_id": job_id,
        "status": status,
        "progress": float(job.meta.get("progress", 0.0)),
        "stage": job.meta.get("stage", status),
        "error": job.exc_info[-2000:] if job.is_failed and job.exc_info else None,
    }
    if job.is_finished:
        body["result"] = job.result
        body["result_url"] = f"/api/jobs/{job_id}/result"
    return body


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    if INLINE_JOBS:
        with _inline_lock:
            j = _inline_jobs.get(job_id)
            if not j or j.get("status") != "finished":
                raise HTTPException(404, "Result is not ready")
            result = j["result"]
    else:
        if _redis is None: raise HTTPException(503, "Render queue is unavailable")
        from rq.job import Job
        try: job = Job.fetch(job_id, connection=_redis)
        except Exception: raise HTTPException(404, "Job not found")
        if not job.is_finished or not job.result:
            raise HTTPException(404, "Result is not ready")
        result = job.result
    p = Path(result["result_path"])
    if not p.exists(): raise HTTPException(404, "Result has expired")
    media = result.get("media_type") or ("image/png" if p.suffix.lower() == ".png" else "image/jpeg")
    return FileResponse(p, media_type=media, filename=result.get("filename", p.name))


@app.get("/api/jobs/{job_id}/preview")
def job_preview(job_id: str):
    """Return the small H.264 playback proxy for a completed video job."""
    if INLINE_JOBS:
        with _inline_lock:
            j=_inline_jobs.get(job_id)
            if not j or j.get("status")!="finished": raise HTTPException(404,"Preview is not ready")
            result=j["result"]
    else:
        if _redis is None: raise HTTPException(503,"Render queue is unavailable")
        from rq.job import Job
        try: job=Job.fetch(job_id,connection=_redis)
        except Exception: raise HTTPException(404,"Job not found")
        if not job.is_finished or not job.result: raise HTTPException(404,"Preview is not ready")
        result=job.result
    pth=result.get("preview_path")
    if not pth: raise HTTPException(404,result.get("preview_error") or "Browser preview was not generated")
    p=Path(pth)
    if not p.exists(): raise HTTPException(404,"Preview has expired")
    return FileResponse(p,media_type="video/mp4")


# Static UI mounted last so /api/* remains authoritative.
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
