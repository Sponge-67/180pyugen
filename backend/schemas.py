from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

class MatchRequest(BaseModel):
    session_id: str
    x: float
    y: float
    decoder: Literal["opencv", "pillow"] = "opencv"

class RenderRequest(BaseModel):
    session_id: str
    left_profile: str
    right_profile: str
    points: dict
    output_height: int = Field(default=1024, ge=128)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    right_shift_x_deg: float = 0.0
    right_shift_y_deg: float = 0.0
    format: Literal["jpeg", "png"] = "jpeg"
    jpeg_quality: int = Field(default=95, ge=1, le=100)
    output_name: str = "LROut.jpg"
    decoder: Literal["opencv", "pillow"] = "opencv"


class VideoMatchRequest(BaseModel):
    session_id: str
    left_frame: int = Field(ge=0)
    right_frame: int = Field(ge=0)
    x: float
    y: float

class VideoRenderRequest(BaseModel):
    session_id: str
    left_profile: str
    right_profile: str
    points: dict
    left_start: int = Field(ge=0)
    left_end: int = Field(ge=0)
    right_start: int = Field(ge=0)
    right_end: int = Field(ge=0)
    output_width: int = Field(default=4096, ge=128)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    codec: str = "auto"
    fps_mode: Literal["kino", "source"] = "kino"
    sampling: Literal["fast", "hq", "native"] = "hq"
    length_policy: Literal["strict", "trim", "repeat_last", "black"] = "strict"
    video_mode: Literal["static", "general"] = "static"
    dynamic_method: Literal["orb_ransac", "dense_flow", "phase"] = "orb_ransac"
    dynamic_smoothing: float = Field(default=0.85, ge=0.0, le=0.995)
    dynamic_horizontal_strength: float = Field(default=0.15, ge=0.0, le=1.0)
    dynamic_max_px: float = Field(default=64.0, ge=1.0, le=512.0)
    encode_quality: Literal["compact", "standard", "high", "very_high", "master", "lossless"] = "high"
    sharpen: float = Field(default=0.15, ge=0.0, le=1.5)
    dejag_mode: Literal["off", "adaptive", "strong"] = "adaptive"
    dejag_strength: float = Field(default=0.45, ge=0.0, le=1.0)
    right_shift_x_deg: float = 0.0
    right_shift_y_deg: float = 0.0
    output_name: str = "VROut.mp4"


class VideoClipRequest(BaseModel):
    session_id: str
    left_start: int = Field(ge=0)
    left_end: int = Field(ge=0)
    right_start: int = Field(ge=0)
    right_end: int = Field(ge=0)
    jpeg_quality: int = Field(default=95, ge=1, le=100)


class AlignmentPreviewRequest(BaseModel):
    session_id: str
    left_profile: str
    right_profile: str
    points: dict
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    right_shift_x_deg: float = 0.0
    right_shift_y_deg: float = 0.0
    preview_height: int = Field(default=384, ge=128, le=768)
    decoder: Literal["opencv", "pillow"] = "opencv"

class VideoAlignmentPreviewRequest(BaseModel):
    session_id: str
    left_profile: str
    right_profile: str
    points: dict
    left_frame: int = Field(ge=0)
    right_frame: int = Field(ge=0)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    right_shift_x_deg: float = 0.0
    right_shift_y_deg: float = 0.0
    preview_height: int = Field(default=384, ge=128, le=768)
