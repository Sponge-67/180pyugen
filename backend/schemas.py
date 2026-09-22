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
    sampling: Literal["fast", "hq", "native"] = "fast"
    length_policy: Literal["strict", "trim", "repeat_last", "black"] = "strict"
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
