#!/usr/bin/env python3
"""180pyugen: clean-room VR180 reconstruction and conversion toolkit.

This single file intentionally contains both the still-image (180Augen-style)
and video (180Kino-style) workflows so the mathematical core cannot silently
drift between two separate applications.  The desktop GUI is only a front end;
the important invariant is that both workflows eventually pass through the same
profile -> ray -> stereo rotation -> equirectangular sampling machinery.

High-level still-image pipeline
-------------------------------
1. Decode the left and right fisheye source images.
2. Resolve each camera/lens profile: 180-degree radius, optical axis, and
   projection function.
3. Convert corresponding image points A/B into unit camera rays.
4. Recover the right-to-left 3-D rotation from those ray correspondences.
5. Generate one 180x180-degree equirectangular ray field per eye.
6. Apply optional zenith correction and the stereo rotation.
7. Project the rays back into each fisheye source and sample the image.
8. Pack left and right eye squares side-by-side.

High-level video pipeline
-------------------------
1. Probe both movies and choose synchronized reference frames.
2. Display those frames immediately in the alignment UI.
3. Acquire A/B correspondences on the synchronized pair.
4. Build the stereo transform and source remap tables once.
5. Reuse those tables for every frame pair; only image sampling and encoding
   occur inside the frame loop.
6. Write the stereo result as MP4/HEVC or another selected codec.

Reverse-engineered/native-oriented behavior is kept separate from convenience
features.  For example, 180Augen uses TM_SQDIFF for point matching whereas
180Kino uses TM_SQDIFF_NORMED; the slower native-style sampler remains
available for parity testing, while optimized OpenCV remap modes are provided
for normal video use.  Exact 180Kino CUDA-kernel parity remains an ongoing
comparison target rather than an assumed fact.

The code is deliberately heavily commented.  Comments document coordinate
conventions, recovered native behavior, performance choices, and places where
our implementation is an engineering extension rather than a native feature.
"""
from __future__ import annotations
import argparse, math, os, threading, time, json, subprocess, shutil, textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

# Optional GUI dependencies (CLI still works without them)
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk, ImageDraw
except ImportError:
    tk = None
    filedialog = messagebox = ttk = None
    Image = ImageTk = ImageDraw = None

def read_image(path: str, decoder="opencv"):
    """Decode JPG/PNG/TIFF/BMP/WebP.

    decoder:
      "opencv" - default; closest to native 180Augen's cv::imread path.
      "pillow" - diagnostic alternative for testing JPEG decoder/IDCT
                 differences against native OpenCV 3.4.1 output.
    """
    if not path or not os.path.isfile(os.fspath(path)):
        return None
    p=os.fspath(path)
    decoder=str(decoder or "opencv").strip().lower()

    def _pillow():
        if Image is None or cv2 is None:
            return None
        try:
            with Image.open(p) as im:
                rgb=np.asarray(im.convert("RGB"))
            return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _opencv():
        if cv2 is None:
            return None
        try:
            data=np.fromfile(p,dtype=np.uint8)
            if data.size:
                img=cv2.imdecode(data,cv2.IMREAD_COLOR)
                if img is not None and img.size:
                    return img
        except Exception:
            pass
        try:
            img=cv2.imread(p,cv2.IMREAD_COLOR)
            if img is not None and img.size:
                return img
        except Exception:
            pass
        return None

    if decoder.startswith("pil"):
        img=_pillow()
        if img is not None:
            return img
        return _opencv()

    img=_opencv()
    if img is not None:
        return img
    return _pillow()

def write_image(path: str, img) -> bool:
    """Write PNG/JPEG/TIFF based on filename extension."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for image I/O")
    ext = Path(path).suffix.lower()
    if ext not in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'):
        raise ValueError(f"Unsupported image output extension: {ext or '<none>'}")
    enc_ext = ext if ext != '.jpeg' else '.jpg'
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if enc_ext == '.jpg' else []
    ok, enc = cv2.imencode(enc_ext, img, params)
    if not ok:
        return False
    enc.tofile(os.fspath(path))
    return True

@dataclass
class Projection:
    mode: int
    k: float = 1.0
    breakpoints: Tuple[float,float,float] = (0.5,1.0,1.3)
    coeffs: Tuple[Tuple[float,float,float], ...] = (
        (-0.03,0.72,0.0),(-0.05,0.75,-0.009),
        (-0.02,0.63,0.083),(-0.37,1.54,-0.506))
    # Native 180Augen and 180Kino both call this projection mode 3, but the
    # supplied 180Kino 2.0 documentation/coefficients use a true quadratic
    # y=a*theta^2+b*theta+c. Keep the formula explicit so the still and video
    # workflows can coexist without silently changing existing Augen profiles.
    custom_formula: str = "augen"

@dataclass
class Profile:
    name: str
    magnification: float
    projection: Projection
    radius_mode: int
    radius: float
    optical_axis_x: float
    optical_axis_y: float
    optical_axis_mode: int

# ---------------------------------------------------------------------------
# Embedded 180Augen profiles
# ---------------------------------------------------------------------------

# The supplied 180Augen.ini is embedded here deliberately.  This means the
# test program no longer depends on an external .ini file.
#
# Each embedded ``d`` list contains the 16 projection values:
#   k,
#   3 projection breakpoints,
#   4 x (A,B,C) coefficient triples.
#
# Radius magnification is a separate optional field because the native INI
# record actually has 17 floating slots.  In most 180Augen profiles this
# leading magnification slot is blank; 180Kino's EM10 profile uses 1.58.
#
# The final 6 values are:
#   radius_mode, radius, projection_mode,
#   optical_axis_x, optical_axis_y, optical_axis_mode.

EMBEDDED_PROFILES = [
    {
        "name": "EM10mkIII & Meike 6.5mm L",
        "d": [
            0.84,
            0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0
        ],
        "q": [3, 2040, 2, 2297, 1743, 1],
    },
    {
        "name": "EM10mkIII & Meike 6.5mm R",
        "d": [
            0.83,
            0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0
        ],
        "q": [3, 2058, 2, 2295, 1682, 1],
    },
    {
        "name": "Canon EOS 5DII & Tokina AT-X 107 DX Fisheye",
        "d": [
            0.84,
            0.5052728,
            1.0288716,
            1.290671,
            -0.0319053368807804,
            0.724831687570841,
            0,
            -0.0500614649420082,
            0.752415125563949,
            -0.00930189068718359,
            -0.0182964300338403,
            0.63014101006576,
            0.0828767457960839,
            -0.371564801930137,
            1.54263916358872,
            -0.506372530630217,
        ],
        "q": [3, 2282, 3, 2787, 1872, 1],
    },
    {
        "name": "5DII & Tokina 17mm",
        "d": [
            0.84,
            0.5052728,
            1.0288716,
            1.290671,
            -0.0319053368807804,
            0.724831687570841,
            0,
            -0.0500614649420082,
            0.752415125563949,
            -0.00930189068718359,
            -0.0182964300338403,
            0.63014101006576,
            0.0828767457960839,
            -0.371564801930137,
            1.54263916358872,
            -0.506372530630217,
        ],
        "q": [3, 3647, 3, 2787, 1872, 0],
    },
    {
        "name": "Laowa",
        "d": [
            0.48,
            0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0
        ],
        "q": [3, 1525, 2, 0, 0, 0],
    },
    {
        "name": "DJI Action2",
        "d": [
            1.00,
            0.5,
            0.61,
            1.0,
            0.110266,
            0.3509746,
            0,
            0.110266,
            0.3509746,
            0,
            0.27851,
            0.1679149,
            0.0490435,
            0.27851,
            0.1679149,
            0.0490435,
        ],
        "q": [3, 3178, 3, 0, 0, 0],
    },
    {
        "name": "EM10mkIII & Meike 6.5mm (180Kino)",
        "magnification": 1.58,
        "d": [0.83, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        "q": [2, 0, 2, 1920, 1080, 0],
    },
    {
        "name": "DJI Mini3 pro(4k)",
        "d": [
            0,0.349,0.7,0.8,
            0.10415024,0.38101724,0,
            0.193344935,0.328066213,0.007615292,
            0.193344935,0.328066213,0.007615292,
            0.193344935,0.328066213,0.007615292,
        ],
        "q": [3,7102,3,0,0,0],
    },
    {
        "name": "DJI Mini2(4k)",
        "d": [
            0,0.349,0.5,0.6,
            0.057654519,0.341915576,0,
            0.268892725,0.198833936,0.02420618,
            0.268892725,0.198833936,0.02420618,
            0.268892725,0.198833936,0.02420618,
        ],
        "q": [3,8610,3,0,0,0],
    },
    {
        "name": "DJI Mini3 pro(2.7k)",
        "d": [
            0,0.349,0.7,0.8,
            0.10415024,0.38101724,0,
            0.193344935,0.328066213,0.007615292,
            0.193344935,0.328066213,0.007615292,
            0.193344935,0.328066213,0.007615292,
        ],
        "q": [3,4971,3,0,0,0],
    },
    {
        "name": "DJI Mini2(2.7k)",
        "d": [
            0,0.349,0.5,0.6,
            0.057654519,0.341915576,0,
            0.268892725,0.198833936,0.02420618,
            0.268892725,0.198833936,0.02420618,
            0.268892725,0.198833936,0.02420618,
        ],
        "q": [3,6099,3,0,0,0],
    },
    {
        "name": "DJI Action5 Fov Boost lens(off mode)",
        "d": [
            0,0.523,1.05,1.2,
            0.007185476,0.674401031,0,
            -0.03688147,0.744128587,-0.02442806,
            -0.28094475,1.28101231,-0.31900604,
            -0.28094475,1.28101231,-0.31900604,
        ],
        "q": [3,1888,3,0,0,0],
    },
    {
        "name": "DJI Action5",
        "d": [
            0,0.42,0.84,1.1,
            0.015042023,0.53812355,0,
            0.084865345,0.48289745,0.010881883,
            0.135385057,0.39012197,0.053148605,
            0.135385057,0.39012197,0.053148605,
        ],
        "q": [3,2766,3,0,0,0],
    },
]



# Profiles originating from the supplied 180Kino.ini.  For mode 3, 180Kino
# uses the documented piecewise quadratic mapping y=a*theta^2+b*theta+c.
KINO_PROFILE_NAMES = {
    "DJI Action2",
    "EM10mkIII & Meike 6.5mm (180Kino)",
    "DJI Mini3 pro(4k)",
    "DJI Mini2(4k)",
    "DJI Mini3 pro(2.7k)",
    "DJI Mini2(2.7k)",
    "DJI Action5 Fov Boost lens(off mode)",
    "DJI Action5",
}

def _dict_to_profile(item):
    """Build a profile from the embedded native profile representation.

    Native files store radius magnification separately from projection k.
    Earlier versions conflated them because the supplied 180Augen profiles
    leave magnification blank.  180Kino's EM10 entry stores 1.58 and 0.83 in
    those two independent slots, which makes the distinction explicit.
    """
    d = item["d"]  # 16 projection values: k,b0,b1,b2, then 4*(A,B,C)
    q = item["q"]
    coeffs = tuple(
        (d[4 + 3*j], d[5 + 3*j], d[6 + 3*j])
        for j in range(4)
    )
    return Profile(
        name=item["name"],
        magnification=float(item.get("magnification", 0.0)),
        projection=Projection(
            mode=int(q[2]),
            k=float(d[0]),
            breakpoints=(float(d[1]), float(d[2]), float(d[3])),
            coeffs=coeffs,
            custom_formula=("kino_quadratic" if item["name"] in KINO_PROFILE_NAMES else "augen"),
        ),
        radius_mode=int(q[0]),
        radius=float(q[1]),
        optical_axis_x=float(q[3]),
        optical_axis_y=float(q[4]),
        optical_axis_mode=int(q[5]),
    )


def embedded_profiles() -> List[Profile]:
    return [_dict_to_profile(x) for x in EMBEDDED_PROFILES]


def parse_profiles(path=None) -> List[Profile]:
    """Parse native 180Augen/180Kino camera profile records.

    Record layout is 17 positional floating slots followed by six integers:
      radius magnification,
      projection k,
      3 interval boundaries,
      4 x (A,B,C),
      radius_mode, radius, projection_mode, axis_x, axis_y, axis_mode.

    Blank floating slots mean zero.  Keeping the leading magnification slot is
    required for 180Kino's EM10 entry (magnification 1.58, k 0.83).
    """
    if path is None:
        return embedded_profiles()

    source_is_kino = "kino" in Path(path).name.lower()
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    out = []
    i = 0

    def is_number(s):
        try:
            float(s.strip())
            return True
        except ValueError:
            return False

    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        name = lines[i].strip()
        i += 1

        slots = []
        while len(slots) < 17 and i < len(lines):
            s = lines[i].strip()
            if not s:
                slots.append(0.0)
                i += 1
            elif is_number(s):
                slots.append(float(s))
                i += 1
            else:
                raise ValueError(
                    f"Unexpected profile name {s!r} while reading "
                    f"17 floating parameter slots for {name!r}"
                )

        while i < len(lines) and not lines[i].strip():
            i += 1

        q = []
        while len(q) < 6 and i < len(lines):
            s = lines[i].strip()
            if not s:
                i += 1
                continue
            if not is_number(s):
                raise ValueError(
                    f"Expected numeric profile integer, got {s!r} "
                    f"after profile {name!r}"
                )
            q.append(int(float(s)))
            i += 1

        if len(slots) != 17 or len(q) != 6:
            raise ValueError(
                f"Incomplete profile {name!r}: "
                f"{len(slots)} floating slots, {len(q)} integers"
            )

        magnification = slots[0]
        d = slots[1:]
        coeffs = tuple(
            (d[4 + 3*j], d[5 + 3*j], d[6 + 3*j])
            for j in range(4)
        )
        out.append(Profile(
            name=name,
            magnification=magnification,
            projection=Projection(
                q[2],
                k=d[0],
                breakpoints=(d[1], d[2], d[3]),
                coeffs=coeffs,
                custom_formula=("kino_quadratic" if source_is_kino else "augen"),
            ),
            radius_mode=q[0],
            radius=float(q[1]),
            optical_axis_x=float(q[3]),
            optical_axis_y=float(q[4]),
            optical_axis_mode=q[5],
        ))
    return out

def profile_by_name(ps:Sequence[Profile], name:str)->Profile:
    for p in ps:
        if p.name==name or p.name.lower()==name.lower(): return p
    m=[p for p in ps if name.lower() in p.name.lower()]
    if len(m)==1:return m[0]
    raise KeyError(f'Profile not found or ambiguous: {name!r}')

def _projection_obj(p):
    # Rendering code sometimes receives a Profile and sometimes its Projection.
    return p.projection if isinstance(p, Profile) else p

def projection_forward(theta, p:Projection, radius=1.0):
    p=_projection_obj(p)
    x=np.asarray(theta,dtype=np.float64); m=p.mode
    if m==0: y=np.sin(x)
    elif m==1: y=math.sqrt(2.0)*np.sin(x/2)
    elif m==2:
        # Native arbitrary-k model: y = A*sin(k*theta), normalized so
        # y(pi/2)=1. 180Augen quantizes A downward to 5 decimal places.
        s90=np.sin(p.k*(math.pi/2.0))
        A=np.floor((100000.0/s90))/100000.0
        y=A*np.sin(p.k*x)
    elif m==3:
        b0,b1,b2=p.breakpoints; y=np.empty_like(x)
        masks=(x<b0,(x>=b0)&(x<b1),(x>=b1)&(x<b2),x>=b2)
        for mask,c in zip(masks,p.coeffs):
            A,B,C=c
            if p.custom_formula=="kino_quadratic":
                y[mask]=A*x[mask]*x[mask]+B*x[mask]+C
            else:
                y[mask]=A*np.sin(x[mask])+B*x[mask]+C
    else: raise ValueError(f'Unsupported projection mode {m}')
    z=radius*y
    return float(z) if np.ndim(theta)==0 else z

def projection_inverse_scalar(y:float,p:Projection,theta_max=math.pi/2,iterations=80)->float:
    p=_projection_obj(p)
    if y<=0:return 0.0
    lo,hi=0.0,theta_max
    if y>=projection_forward(theta_max,p): return hi
    for _ in range(iterations):
        mid=(lo+hi)/2
        if projection_forward(mid,p)<y: lo=mid
        else: hi=mid
    return (lo+hi)/2

def projection_inverse(y,p,theta_max=math.pi/2):
    p=_projection_obj(p)
    a=np.asarray(y,dtype=np.float64)
    if p.mode==0: z=np.arcsin(np.clip(a,-1,1))
    elif p.mode==1: z=2*np.arcsin(np.clip(a/math.sqrt(2.0),-1,1))
    elif p.mode==2:
        s90=np.sin(p.k*(math.pi/2.0))
        A=np.floor((100000.0/s90))/100000.0
        z=(1.0/p.k)*np.arcsin(np.clip(a/A,-1,1))
    else:
        flat=a.ravel(); z=np.array([projection_inverse_scalar(float(v),p,theta_max) for v in flat]).reshape(a.shape)
    return float(z) if np.ndim(y)==0 else z

def gopro12_max_lens_mod2_profile(width: int,
                                   height: int,
                                   fov_deg: float = 177.0) -> Profile:
    """
    Construct an approximate GoPro HERO12 + Max Lens Mod 2.0 profile.

    The official GoPro specification gives a maximum 177-degree FOV for
    Max HyperView at 4K60, but does not publish the optical calibration
    coefficients needed to make a pixel-accurate 180Augen profile.

    We therefore model the image as an equidistant projection:

        r = f * theta

    and choose f so that the supplied full FOV reaches the image's radial
    edge:

        f = R / (FOV/2)

    For 177 degrees, theta_edge = 88.5 degrees.

    This is useful for testing the pipeline, NOT a calibrated production
    profile.
    """
    theta_edge = math.radians(fov_deg / 2.0)

    # For the 8:7 HERO12 sensor/frame, using half the smaller dimension
    # would be a conservative circular-image radius.  The actual Max Lens
    # Mod output is digitally warped/cropped depending on the selected
    # GoPro digital lens, so this is intentionally configurable.
    radius = min(width, height) / 2.0

    # Piecewise A*sin(theta)+B*theta+C approximating r/R = theta/theta_edge.
    # A=0, B=1/theta_edge, C=0 is exact for the model.
    b = 1.0 / theta_edge

    return Profile(
        name="GoPro HERO12 Black + Max Lens Mod 2.0 (177deg equidistant approx)",
        magnification=1.0,
        projection=Projection(
            mode=3,
            breakpoints=(
                math.pi / 4.0,
                3.0 * math.pi / 8.0,
                7.0 * math.pi / 16.0,
            ),
            coeffs=(
                (0.0, b, 0.0),
                (0.0, b, 0.0),
                (0.0, b, 0.0),
                (0.0, b, 0.0),
            ),
        ),
        radius_mode=3,
        radius=radius,
        optical_axis_x=width / 2.0,
        optical_axis_y=height / 2.0,
        optical_axis_mode=1,
    )


GOPRO_CAL_L_NAME = "GoPro HERO12 + Max Lens Mod 2.0 L (calibrated)"
GOPRO_CAL_R_NAME = "GoPro HERO12 + Max Lens Mod 2.0 R (calibrated)"

# Calibration reference: supplied 1792x2048 GoPro stereo pair.
# Fitted physical image-circle edge:
#   L center ~= (891.75, 1020.03), edge radius ~= 899.93 px
#   R center ~= (896.57, 1024.94), edge radius ~= 898.98 px
#
# With the 177-degree full-FOV constraint and plumb-line fitting to straight
# room edges, the arbitrary-k model converges near k=0.652.  Converted to the
# 180Augen convention where the configured radius means theta=90 degrees:
#   L radius ~= 909.51 px
#   R radius ~= 908.55 px
GOPRO_CAL_BASE_W = 1792.0
GOPRO_CAL_BASE_H = 2048.0
GOPRO_CAL_K = 0.652

def gopro12_max_lens_mod2_calibrated_profile(side: str,
                                             width: int = 1792,
                                             height: int = 2048) -> Profile:
    side = str(side).upper()
    if side not in ("L", "R"):
        raise ValueError("side must be 'L' or 'R'")

    sx = float(width) / GOPRO_CAL_BASE_W
    sy = float(height) / GOPRO_CAL_BASE_H
    s = 0.5 * (sx + sy)

    if side == "L":
        cx0, cy0, r0 = 891.75, 1020.03, 909.51
        name = GOPRO_CAL_L_NAME
    else:
        cx0, cy0, r0 = 896.57, 1024.94, 908.55
        name = GOPRO_CAL_R_NAME

    return Profile(
        name=name,
        magnification=1.0,
        projection=Projection(
            mode=2,
            k=GOPRO_CAL_K,
            breakpoints=(0.0, 0.0, 0.0),
            coeffs=((0.0,0.0,0.0),)*4,
        ),
        radius_mode=3,
        radius=r0 * s,
        optical_axis_x=cx0 * sx,
        optical_axis_y=cy0 * sy,
        optical_axis_mode=1,
    )



def gopro_calibration_compatibility(width:int,height:int)->tuple[bool,str]:
    """Check whether the image-calibrated GoPro geometry can be transferred safely.

    The current HERO12 + Max Lens Mod 2.0 calibration was measured from a
    1792x2048 portrait source.  A proportional resize is safe: optical centre,
    radius and A/B coordinates scale consistently.  A different aspect ratio
    usually means a different GoPro digital-lens crop/orientation/stabilisation
    mode, so silently treating it as the still calibration can produce a very
    different projection.
    """
    w=max(1,int(width)); h=max(1,int(height))
    base=GOPRO_CAL_BASE_W/GOPRO_CAL_BASE_H
    aspect=w/h
    rel=abs(aspect/base-1.0)
    ok=rel <= 0.02
    if ok:
        return True, f"GoPro calibration aspect compatible ({w}x{h}); profile is scaled proportionally."
    rotated=abs(aspect-(1.0/base))/(1.0/base) <= 0.02
    if rotated:
        why="frame aspect looks like a 90-degree rotation of the calibration source"
    else:
        why=f"frame aspect {aspect:.4f} differs from calibration aspect {base:.4f}"
    return False, ("GoPro still-image calibration is not geometry-compatible with this video frame: "
                   +why+". Keep the GoPro lens profile only as a starting point and pick fresh A/B "
                   "points from this video's synchronized reference frames; a video-specific lens/crop "
                   "calibration may be required.")


def scale_gopro_reference_point(pt,width:int,height:int):
    """Scale a point from the *specific supplied calibration still pair*.

    This helper exists for regression tests and that exact scene only.  A/B are
    scene correspondence points, not lens calibration constants, so the video
    preset must never inject these coordinates into unrelated footage.
    """
    return (float(pt[0])*float(width)/GOPRO_CAL_BASE_W,
            float(pt[1])*float(height)/GOPRO_CAL_BASE_H)


def calculate_radius(p:Profile,w:int,h:int)->float:
    if p.radius_mode==0:return min(w,h)/2
    if p.radius_mode==1:return math.hypot(w,h)/2
    if p.radius_mode==2:return h/2*p.magnification
    if p.radius_mode==3:return p.radius
    raise ValueError('bad radius mode')

def calculate_optical_axis(p:Profile,w:int,h:int):
    return (w/2,h/2) if p.optical_axis_mode==0 else (p.optical_axis_x,p.optical_axis_y)

def exe_projection_radius(theta, p, radius):
    """Map incident angle to source radius using the selected lens model.

    Radius magnification is handled only by calculate_radius(); projection k
    is a separate native parameter.
    """
    pp = p.projection if isinstance(p, Profile) else p
    return radius * projection_forward(theta, pp, 1.0)


def exe_spherical_direction(row, col, n):
    """Per-eye 180°x180° equirectangular direction used by 180Augen.

    lon spans -90..+90 degrees left-to-right.
    lat spans +90..-90 degrees top-to-bottom.
    Camera/world frame:
      +X = image right
      +Y = image up
      +Z = optical axis / forward
    """
    lon = (float(col) / float(n)) * math.pi - (math.pi / 2.0)
    lat = (math.pi / 2.0) - (float(row) / float(n)) * math.pi
    cl = math.cos(lat)
    return np.array([
        cl * math.sin(lon),
        math.sin(lat),
        cl * math.cos(lon)
    ], dtype=np.float64)


def exe_sample_camera_directions(img, p, dirs):
    """Map unit directions into a fisheye image using the EM10/180Augen frame.

    +Z = optical axis, +X = image right, +Y = image up.
    The source raster has +y downward, therefore source Y subtracts the
    Cartesian +Y component.  Native behavior relies on source-raster bounds
    rather than a separate 90-degree hemisphere mask.
    """
    h,w=img.shape[:2]
    d=np.asarray(dirs,dtype=np.float64)

    z=np.clip(d[:,2],-1.0,1.0)
    theta=np.arccos(z)
    plane=np.hypot(d[:,0],d[:,1])

    r=exe_projection_radius(theta,p,calculate_radius(p,w,h))
    cx,cy=calculate_optical_axis(p,w,h)

    px=np.full(len(d),cx,dtype=np.float64)
    py=np.full(len(d),cy,dtype=np.float64)
    nz=plane>1e-14
    px[nz]=cx+r[nz]*(d[nz,0]/plane[nz])
    py[nz]=cy-r[nz]*(d[nz,1]/plane[nz])

    # Native 180Augen does not appear to apply a separate theta<=90deg mask
    # here.  It projects the direction into source-image coordinates and then
    # rejects only coordinates outside the source raster.  This matters on the
    # transformed right eye: some valid source pixels lie slightly outside the
    # nominal calibrated 180-degree radius.
    # Important native edge behavior:
    # 180Augen does NOT reject the floating-point coordinate before converting
    # it to an integer.  It first executes cvttsd2si (truncate toward zero),
    # then bounds-checks each of the four integer neighbors separately.
    #
    # Therefore coordinates such as x=-0.25 become x0=0 with fx=-0.25 and
    # still participate in interpolation.  Rejecting px<0/py<0 here, as v16
    # did, creates a one-pixel-thick black fringe that the native EXE does not.
    valid=np.isfinite(px)&np.isfinite(py)

    out=bilinear_sample(img,px,py,valid)
    return out


def render_stereo_exe_geometry(left,right,lp,rp,left_a,left_b,right_a,right_b,output_height=1024,roll=0,pitch=0,yaw=0):
    """Reconstruct 180Augen's side-by-side VR180 renderer.

    Each eye is a 180°x180° equirectangular square.  The left eye defines the
    reference/world orientation.  The recovered two-point frame rotation maps
    right-camera rays into that reference frame, so world rays are converted
    back into right-camera rays with row-vector multiplication by R.
    """
    n=int(output_height)
    la=pixel_to_ray(*left_a,lp,left.shape[1],left.shape[0])
    lb=pixel_to_ray(*left_b,lp,left.shape[1],left.shape[0])
    ra=pixel_to_ray(*right_a,rp,right.shape[1],right.shape[0])
    rb=pixel_to_ray(*right_b,rp,right.shape[1],right.shape[0])
    R=stereo_rotation(la,lb,ra,rb)
    Z=zenith_rotation(roll,pitch,yaw)
    yy,xx=np.indices((n,n),dtype=np.float64)
    d=np.empty((n*n,3),dtype=np.float64)
    lon=xx.reshape(-1)*(math.pi/n) - (math.pi/2.0)
    lat=(math.pi/2.0) - yy.reshape(-1)*(math.pi/n)
    cl=np.cos(lat)
    d[:,0]=cl*np.sin(lon)
    d[:,1]=np.sin(lat)
    d[:,2]=cl*np.cos(lon)
    # Common output frame -> camera frames.  Keep the left frame as reference.
    dl=d@Z.T
    dr=(d@Z.T)@R
    L=exe_sample_camera_directions(left,lp,dl).reshape(n,n,3)
    RR=exe_sample_camera_directions(right,rp,dr).reshape(n,n,3)
    return np.concatenate([L,RR],axis=1)


def pixel_to_ray(x,y,p,w,h):
    """Convert a fisheye source pixel to the 180Augen camera frame.

    +Z = optical axis, +X = image right, +Y = image up.
    """
    r=calculate_radius(p,w,h)
    cx,cy=calculate_optical_axis(p,w,h)
    dx=x-cx
    dy=y-cy
    rho=math.hypot(dx,dy)
    if rho<1e-12:
        return np.array([0.,0.,1.])

    theta=projection_inverse(rho/r,p)
    st=math.sin(theta)
    return np.array([
        (dx/rho)*st,
        -(dy/rho)*st,
        math.cos(theta)
    ],dtype=float)


def ray_to_pixel(ray,p,w,h):
    ray=np.asarray(ray,dtype=float)
    ray/=np.linalg.norm(ray)
    theta=math.acos(np.clip(ray[2],-1,1))
    rr=projection_forward(theta,p,calculate_radius(p,w,h))
    plane=math.hypot(ray[0],ray[1])
    cx,cy=calculate_optical_axis(p,w,h)
    if plane<1e-15:
        return cx,cy
    return cx+rr*ray[0]/plane, cy-rr*ray[1]/plane


def normalize(v):
    v=np.asarray(v,dtype=float); return v/np.linalg.norm(v)

def make_frame(a,b):
    """Stable orthonormal basis from the two selected correspondence rays."""
    e1=normalize(a)
    q=np.asarray(b,dtype=float)-np.dot(b,e1)*e1
    if np.linalg.norm(q)<1e-10:raise ValueError('A/B rays are nearly collinear')
    e2=normalize(q)
    e3=normalize(np.cross(e1,e2))
    e2=normalize(np.cross(e3,e1))
    return np.column_stack((e1,e2,e3))

def _rotation_from_to(a,b):
    """Shortest proper rotation taking unit vector a onto unit vector b.

    This is the Rodrigues form used by the native EXE:
        v = a x b
        c = a . b
        R = I + [v]x + [v]x^2/(1+c)
    """
    a=normalize(a); b=normalize(b)
    v=np.cross(a,b)
    c=float(np.dot(a,b))
    s2=float(np.dot(v,v))
    if s2 < 1e-24:
        if c > 0:
            return np.eye(3,dtype=float)
        # Rare antipodal fallback: choose a stable perpendicular axis.
        axis=np.cross(a,np.array([1.,0.,0.]))
        if np.linalg.norm(axis)<1e-8:
            axis=np.cross(a,np.array([0.,1.,0.]))
        axis=normalize(axis)
        K=np.array([[0.,-axis[2],axis[1]],
                    [axis[2],0.,-axis[0]],
                    [-axis[1],axis[0],0.]],dtype=float)
        return np.eye(3)+2.0*(K@K)

    K=np.array([[0.,-v[2],v[1]],
                [v[2],0.,-v[0]],
                [-v[1],v[0],0.]],dtype=float)
    return np.eye(3)+K+(K@K)/(1.0+c)


def _axis_rotation_from_cs(axis,c,s):
    """Rodrigues rotation around a unit axis from precomputed cos/sin."""
    a=normalize(axis)
    K=np.array([[0.,-a[2],a[1]],
                [a[2],0.,-a[0]],
                [-a[1],a[0],0.]],dtype=float)
    return c*np.eye(3)+(1.0-c)*np.outer(a,a)+s*K


def stereo_rotation(left_a,left_b,right_a,right_b):
    """Native 180Augen two-point right->left rotation.

    Reverse-engineered from the scalar block around 0x140002A79-0x140002EB4.

    Stage 1:
      Rotate right A exactly onto left A with the shortest Rodrigues rotation.

    Stage 2:
      Rotate around the already-aligned A axis to bring B into correspondence.
      The native EXE assumes the A-B angular separation is the same in both
      cameras.  Instead of independently normalizing both projected B vectors,
      it uses the right-pair A-B cosine for both sides:

          d    = dot(R1*right_B, left_A)
          c2   = (dot(R1*right_B, left_B) - d*d) / (1 - d*d)
          |s2| = sqrt(1 - c2*c2)

      The sign is selected from:
          left_A . ((R1*right_B) x left_B)

    This tiny detail differs from the earlier make_frame() approximation by
    about 0.039 degrees for the canonical EM10 test pair, which is enough to
    explain the remaining right-eye mismatch against native LROut.jpg.
    """
    la=normalize(left_a)
    lb=normalize(left_b)
    ra=normalize(right_a)
    rb=normalize(right_b)

    R1=_rotation_from_to(ra,la)
    rb1=R1@rb

    d=float(np.dot(rb1,la))
    denom=1.0-d*d
    if denom < 1e-15:
        return R1

    c2=(float(np.dot(rb1,lb))-d*d)/denom
    c2=float(np.clip(c2,-1.0,1.0))
    s2=math.sqrt(max(0.0,1.0-c2*c2))

    triple=float(np.dot(la,np.cross(rb1,lb)))
    if triple < 0.0:
        s2=-s2

    R2=_axis_rotation_from_cs(la,c2,s2)
    return R2@R1


def Rx(a):
    c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a):
    c,s=math.cos(a),math.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a):
    c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def zenith_rotation(roll_deg,pitch_deg,yaw_deg):
    return Rz(math.radians(yaw_deg))@Ry(math.radians(pitch_deg))@Rx(math.radians(roll_deg))

def equirect_direction(u,v,w,h):
    lon=(u/w-.5)*2*math.pi; lat=(.5-v/h)*math.pi; cl=np.cos(lat)
    return np.stack((cl*np.sin(lon),np.sin(lat),cl*np.cos(lon)),-1)

def bilinear_sample(img,x,y,valid):
    """Native 180Augen bilinear sampler.

    Reverse-engineered from 0x14000330A..0x1400035B4:
      * source integer coordinates use truncation toward zero;
      * each of the four neighbors is bounds-checked independently;
      * an out-of-bounds neighbor contributes black rather than invalidating
        the entire destination pixel;
      * horizontal interpolation is performed first, then vertical;
      * final channel values are truncated toward zero with cvttsd2si.

    This differs subtly from cv2.remap and from the earlier Python sampler at
    source-image edges.
    """
    h,w=img.shape[:2]
    x=np.asarray(x,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    valid=np.asarray(valid,dtype=bool)

    out=np.zeros((len(x),img.shape[2]),dtype=np.uint8)
    idx=np.nonzero(valid & np.isfinite(x) & np.isfinite(y))[0]
    if len(idx)==0:
        return out

    xx=x[idx]
    yy=y[idx]

    # cvttsd2si = truncation toward zero. For normal in-image coordinates this
    # is the same as floor, but use trunc explicitly to mirror the executable.
    x0=np.trunc(xx).astype(np.int64)
    y0=np.trunc(yy).astype(np.int64)
    fx=xx-x0
    fy=yy-y0

    n=len(idx)
    p00=np.zeros((n,3),dtype=np.float64)
    p10=np.zeros((n,3),dtype=np.float64)
    p01=np.zeros((n,3),dtype=np.float64)
    p11=np.zeros((n,3),dtype=np.float64)

    m00=(x0>=0)&(x0<w)&(y0>=0)&(y0<h)
    if np.any(m00):
        p00[m00]=img[y0[m00],x0[m00]].astype(np.float64)

    m10=(x0>=0)&(x0<w-1)&(y0>=0)&(y0<h)
    if np.any(m10):
        p10[m10]=img[y0[m10],x0[m10]+1].astype(np.float64)

    m01=(x0>=0)&(x0<w)&(y0>=0)&(y0<h-1)
    if np.any(m01):
        p01[m01]=img[y0[m01]+1,x0[m01]].astype(np.float64)

    m11=(x0>=0)&(x0<w-1)&(y0>=0)&(y0<h-1)
    if np.any(m11):
        p11[m11]=img[y0[m11]+1,x0[m11]+1].astype(np.float64)

    # Match the executable's arithmetic order:
    # top    = p10*fx + p00*(1-fx)
    # bottom = p11*fx + p01*(1-fx)
    # value  = top*(1-fy) + bottom*fy
    wx0=1.0-fx
    wy0=1.0-fy
    top=p10*fx[:,None] + p00*wx0[:,None]
    bottom=p11*fx[:,None] + p01*wx0[:,None]
    val=top*wy0[:,None] + bottom*fy[:,None]

    # Native code uses cvttsd2si on each channel before writing the byte.
    q=np.trunc(val).astype(np.int64)
    q=np.clip(q,0,255).astype(np.uint8)
    out[idx]=q
    return out


def sample_camera_directions(img,p,dirs):
    h,w=img.shape[:2]; z=np.clip(dirs[:,2],-1,1); valid=z>=0
    theta=np.arccos(z); r=projection_forward(theta,p,calculate_radius(p,w,h)); xy=np.hypot(dirs[:,0],dirs[:,1]); cx,cy=calculate_optical_axis(p,w,h)
    px=np.full(len(dirs),cx); py=np.full(len(dirs),cy); nz=xy>1e-14
    px[nz]=cx+r[nz]*dirs[nz,0]/xy[nz]; py[nz]=cy+r[nz]*dirs[nz,1]/xy[nz]
    valid &= (px>=0)&(py>=0)&(px<w-1)&(py<h-1)
    return bilinear_sample(img,px,py,valid)

def render_fisheye(img,p,output_height=512,rotation=None):
    h,w=img.shape[:2]; oh=int(output_height); ow=2*oh; yy,xx=np.indices((oh,ow),float); d=equirect_direction(xx,yy,ow,oh).reshape(-1,3)
    if rotation is not None:d=d@rotation.T
    return np.clip(sample_camera_directions(img,p,d).reshape(oh,ow,3),0,255).astype(np.uint8)

def render_stereo_vr180(left,right,lp,rp,left_a,left_b,right_a,right_b,output_height=512,roll=0,pitch=0,yaw=0):
    lh,lw=left.shape[:2]; rh,rw=right.shape[:2]
    la=pixel_to_ray(*left_a,lp,lw,lh); lb=pixel_to_ray(*left_b,lp,lw,lh); ra=pixel_to_ray(*right_a,rp,rw,rh); rb=pixel_to_ray(*right_b,rp,rw,rh)
    R=stereo_rotation(la,lb,ra,rb); Z=zenith_rotation(roll,pitch,yaw)
    oh=int(output_height); ow=2*oh; yy,xx=np.indices((oh,ow),float); world=equirect_direction(xx,yy,ow,oh).reshape(-1,3)
    out=np.zeros((len(world),3),float); dl=world@Z.T; lm=world[:,0]>=0
    out[lm]=sample_camera_directions(left,lp,dl[lm])
    dr=(world[~lm]@Z.T)@R; out[~lm]=sample_camera_directions(right,rp,dr)
    return np.clip(out.reshape(oh,ow,3),0,255).astype(np.uint8)

def template_match_point(left, right, x, y, template_size=20):
    """Reproduce 180Augen's native Get A/B correspondence matcher.

    Reverse-engineered from 180Augen.exe:
      * click is converted to full-resolution source coordinates first
      * template is exactly 20x20, centered at the clicked source coordinate
      * cv::matchTemplate method = 0 (TM_SQDIFF)
      * cv::minMaxLoc minimum location is used
      * +10,+10 converts the matching rectangle's upper-left to its center

    ``template_size`` is retained for CLI/API compatibility but the native
    algorithm always uses 20.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required")
    if left is None or right is None:
        raise ValueError("Both images are required")

    lh,lw = left.shape[:2]
    rh,rw = right.shape[:2]
    ix = int(x)
    iy = int(y)

    half = 10
    if ix < half or iy < half or ix > lw-half or iy > lh-half:
        raise ValueError("Point is too close to the left-image edge")

    # Native ROI: x-10, y-10, width=20, height=20.
    templ = left[iy-half:iy+half, ix-half:ix+half]
    if templ.shape[0] != 20 or templ.shape[1] != 20:
        raise ValueError("Could not form native 20x20 matching template")

    res = cv2.matchTemplate(right, templ, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    rx = int(min_loc[0]) + half
    ry = int(min_loc[1]) + half

    if rx < half or ry < half or rx > rw-half or ry > rh-half:
        raise ValueError("Matched point is too close to the right-image edge")

    # Existing callers expect a 'score'. For SQDIFF, lower is better.
    return float(rx), float(ry), float(min_val)


def template_match_point_kino(left, right, x, y, template_size=20):
    """Reproduce 180Kino 2.0's Get A/B matcher.

    Disassembly of 180Kino2.exe shows:
      * native 20x20 template ROI centered on the clicked left coordinate
      * cv::matchTemplate method = 1 (TM_SQDIFF_NORMED)
      * cv::minMaxLoc minimum location
      * +10,+10 converts match rectangle origin to point center

    This intentionally differs from 180Augen, which uses method 0
    (TM_SQDIFF) in the still-image path.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required")
    if left is None or right is None:
        raise ValueError("Both images are required")

    lh,lw = left.shape[:2]
    ix = int(x)
    iy = int(y)
    half = 10
    if ix < half or iy < half or ix > lw-half or iy > lh-half:
        raise ValueError("Point is too close to the left-image edge")

    templ = left[iy-half:iy+half, ix-half:ix+half]
    if templ.shape[0] != 20 or templ.shape[1] != 20:
        raise ValueError("Could not form native 20x20 matching template")

    res = cv2.matchTemplate(right, templ, cv2.TM_SQDIFF_NORMED)
    min_val, _, min_loc, _ = cv2.minMaxLoc(res)
    rx = int(min_loc[0]) + half
    ry = int(min_loc[1]) + half
    return float(rx), float(ry), float(min_val)

def native_preview_geometry(width, height, side="left"):
    """Return the exact native preview scale/dimensions recovered from the EXE.

    Large left preview: max dimension -> 1024.
    Small right preview: max dimension -> 512.
    The other dimension is truncated toward zero after scaling.
    """
    limit = 1024.0 if str(side).lower().startswith("l") else 512.0
    m = max(int(width), int(height))
    if m <= 0:
        raise ValueError("Invalid image dimensions")
    scale = limit / float(m)
    if width >= height:
        pw = int(limit)
        ph = int(height * scale)
    else:
        ph = int(limit)
        pw = int(width * scale)
    return scale, pw, ph

def native_display_to_source(x, y, width, height):
    """Map a click in the 1024-max-dimension left preview to source pixels."""
    scale, _, _ = native_preview_geometry(width, height, "left")
    return int(float(x) / scale), int(float(y) / scale)

def native_source_to_display(x, y, width, height, side="left"):
    """Map source pixels into the native left/right preview coordinate system."""
    scale, _, _ = native_preview_geometry(width, height, side)
    return int(float(x) * scale), int(float(y) * scale)

def make_synthetic_fisheye(p,w=800,h=800):
    if cv2 is None:raise RuntimeError('OpenCV required')
    img=np.zeros((h,w,3),np.uint8); cx,cy=calculate_optical_axis(p,w,h); R=calculate_radius(p,w,h); yy,xx=np.indices((h,w),float); dx,dy=xx-cx,yy-cy; rho=np.hypot(dx,dy); nr=rho/R; lim=projection_forward(math.pi/2,p.projection); valid=nr<=lim
    theta=np.zeros_like(nr); theta[valid]=projection_inverse(nr[valid],p.projection); az=np.arctan2(dy,dx)
    c0=.5+.5*np.sin(8*az)*np.cos(5*theta); c1=.5+.5*np.cos(6*az+2*theta); c2=.5+.5*np.sin(10*theta)
    img[...,0]=np.where(valid,c0*255,0).astype(np.uint8); img[...,1]=np.where(valid,c1*255,0).astype(np.uint8); img[...,2]=np.where(valid,c2*255,0).astype(np.uint8)
    for t in np.linspace(.2,math.pi/2,7):
        ring=np.abs(rho-projection_forward(t,p.projection,R))<2; img[ring]=(255,255,255)
    for a in np.linspace(-math.pi,math.pi,16,endpoint=False):
        line=np.abs(np.angle(np.exp(1j*(az-a))))<.012; img[valid&line]=(255,255,255)
    for px,py in [(cx,cy),(cx+R*.3,cy),(cx-R*.3,cy),(cx,cy+R*.3),(cx,cy-R*.3)]: cv2.circle(img,(round(px),round(py)),7,(255,0,255),-1)
    return img


# ---------------------------------------------------------------------------
# 180Kino-style video workflow
# ---------------------------------------------------------------------------

def video_probe(path: str) -> dict:
    """Return basic OpenCV metadata. Frame numbering is zero-based.

    Only inexpensive container/stream metadata is read here; no full movie is
    decoded.  The returned FPS is preserved as a float because cameras often
    report NTSC-derived rates such as 60000/1001 rather than an integer 60.
    Conversion can later choose whether to preserve that value or emulate the
    native 180Kino rounded-output timing behavior.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video support")
    cap = cv2.VideoCapture(os.fspath(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
        fourcc_i = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc = "".join(chr((fourcc_i >> (8*i)) & 0xff) for i in range(4)).rstrip("\x00")
        return {"path":os.fspath(path),"fps":fps,"frame_count":count,
                "width":width,"height":height,"fourcc":fourcc}
    finally:
        cap.release()


def read_video_frame(path: str, frame_no: int):
    """Random-access one zero-based frame using OpenCV VideoCapture.

    Reference-frame reads are deliberately isolated from the streaming render
    loop.  They are used by the GUI/web preview, A/B matching, and clipping
    tools.  The conversion loop opens each movie once and reads sequentially,
    which avoids repeated decoder seek/setup overhead.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video support")
    frame_no = int(frame_no)
    if frame_no < 0:
        raise ValueError("Frame number must be >= 0")
    cap = cv2.VideoCapture(os.fspath(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise IndexError(f"Could not read frame {frame_no} from {path}")
        return frame
    finally:
        cap.release()


def clip_video_jpegs(left_path, right_path, out_dir,
                     left_start, left_end, right_start, right_end,
                     jpeg_quality=95, progress=None):
    """Extract JPEGs as L<frame>.jpg / R<frame>.jpg, like 180Kino.

    Left and right ranges may differ in length because the clipping tool is
    primarily a synchronization aid.  Final stereo conversion is stricter: it
    requires equal numbers of synchronized frame pairs.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video support")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranges = [("L",left_path,int(left_start),int(left_end)),
              ("R",right_path,int(right_start),int(right_end))]
    total = sum(max(0,e-s+1) for _,_,s,e in ranges)
    done = 0
    written = []
    for prefix,path,start,end in ranges:
        if end < start:
            raise ValueError(f"{prefix} end frame must be >= start frame")
        cap = cv2.VideoCapture(os.fspath(path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for frame_no in range(start,end+1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not read {prefix} frame {frame_no}")
                fn = out_dir / f"{prefix}{frame_no}.jpg"
                ok2, enc = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,int(jpeg_quality)])
                if not ok2:
                    raise RuntimeError(f"Could not encode {fn.name}")
                enc.tofile(os.fspath(fn))
                written.append(str(fn))
                done += 1
                if progress:
                    progress(done,total,f"Clipping {prefix} frame {frame_no}")
        finally:
            cap.release()
    return written


def _directions_to_source_coordinates(shape, p, dirs):
    h,w = int(shape[0]), int(shape[1])
    d = np.asarray(dirs,dtype=np.float64)
    z = np.clip(d[:,2],-1.0,1.0)
    theta = np.arccos(z)
    plane = np.hypot(d[:,0],d[:,1])
    r = exe_projection_radius(theta,p,calculate_radius(p,w,h))
    cx,cy = calculate_optical_axis(p,w,h)
    px = np.full(len(d),cx,dtype=np.float64)
    py = np.full(len(d),cy,dtype=np.float64)
    nz = plane > 1e-14
    px[nz] = cx + r[nz]*(d[nz,0]/plane[nz])
    py[nz] = cy - r[nz]*(d[nz,1]/plane[nz])
    return px,py



def make_alignment_overlay_rgb(left_rgb,right_rgb,alpha=0.5,mode="negative",dx=0.0,dy=0.0):
    """Build a stereo registration diagnostic view.

    mode="negative" mirrors BorisFX/Silhouette Stereo Align: one view is
    inverted and mixed with the other, so matched structure tends toward flat
    mid-gray. dx/dy shift the RIGHT view in source-image pixels for preview
    only; they do not alter the calibrated 3-D render.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required")
    L=np.asarray(left_rgb,dtype=np.uint8)
    R=np.asarray(right_rgb,dtype=np.uint8)
    h,w=R.shape[:2]
    if L.shape[:2] != (h,w):
        L=cv2.resize(L,(w,h),interpolation=cv2.INTER_LANCZOS4)
    if abs(float(dx))>1e-12 or abs(float(dy))>1e-12:
        M=np.array([[1.0,0.0,float(dx)],[0.0,1.0,float(dy)]],np.float32)
        R=cv2.warpAffine(R,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    a=max(0.0,min(1.0,float(alpha)))
    m=str(mode or 'negative').strip().lower()
    if m in ('negative','negative align','invert','emboss'):
        out=L.astype(np.float32)*a + (255.0-R.astype(np.float32))*(1.0-a)
        return np.clip(out,0,255).astype(np.uint8)
    if m in ('difference','diff'):
        return cv2.absdiff(L,R)
    if m in ('anaglyph','red-cyan','red/cyan'):
        out=R.copy()
        out[:,:,0]=L[:,:,0]  # RGB red from left, cyan from right
        return out
    if m in ('perceptual fuse','perceptual','fused'):
        # Best monoscopic proxy of the final stereo result on a normal 2-D
        # display: the already-projected left/right eyes are fused into one
        # image after the candidate right-eye correction is applied.
        return cv2.addWeighted(L,0.5,R,0.5,0.0)
    return cv2.addWeighted(R,1.0-a,L,a,0.0)


def apply_right_eye_trim_sbs(out,right_shift_x_deg=0.0,right_shift_y_deg=0.0,
                             interpolation=None):
    """Apply optional final right-eye convergence/vertical trim in VR degrees.

    Each eye spans 180 degrees across its square.  Zero values are a no-op.
    This is intentionally separate from source-image overlay offsets.
    """
    if cv2 is None:
        return out
    xdeg=float(right_shift_x_deg); ydeg=float(right_shift_y_deg)
    if abs(xdeg)<1e-12 and abs(ydeg)<1e-12:
        return out
    h,w=out.shape[:2]
    eye=w//2
    if eye<=0:
        return out
    interp=cv2.INTER_LINEAR if interpolation is None else int(interpolation)
    dx=xdeg*eye/180.0; dy=ydeg*h/180.0
    M=np.array([[1.0,0.0,dx],[0.0,1.0,dy]],np.float32)
    R=cv2.warpAffine(out[:,eye:],M,(w-eye,h),flags=interp,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return np.concatenate([out[:,:eye],R],axis=1)


# A video frame changes every iteration, but camera geometry does not.  The
# VideoRemapPlan is therefore the performance-critical boundary between setup
# and the hot frame loop.  It stores source-coordinate maps for both eyes plus
# the stereo/output dimensions.  Fast mode can convert the float maps into
# OpenCV's compact fixed-point representation once, saving both memory traffic
# and conversion work on every frame.
@dataclass
class VideoRemapPlan:
    output_height: int
    left_x: Optional[np.ndarray]
    left_y: Optional[np.ndarray]
    right_x: Optional[np.ndarray]
    right_y: Optional[np.ndarray]
    left_map1: Optional[np.ndarray] = None
    left_map2: Optional[np.ndarray] = None
    right_map1: Optional[np.ndarray] = None
    right_map2: Optional[np.ndarray] = None


def build_video_remap_plan(left_shape,right_shape,lp,rp,
                           left_a,left_b,right_a,right_b,
                           output_height=2048,roll=0.0,pitch=0.0,yaw=0.0,
                           block_rows=64,progress=None,map_dtype=np.float32):
    """Precompute fixed source-coordinate maps once for all video frames."""
    n = int(output_height)
    if n < 64:
        raise ValueError("Output height must be >= 64")
    lh,lw = int(left_shape[0]),int(left_shape[1])
    rh,rw = int(right_shape[0]),int(right_shape[1])
    la=pixel_to_ray(*left_a,lp,lw,lh); lb=pixel_to_ray(*left_b,lp,lw,lh)
    ra=pixel_to_ray(*right_a,rp,rw,rh); rb=pixel_to_ray(*right_b,rp,rw,rh)
    R=stereo_rotation(la,lb,ra,rb); Z=zenith_rotation(roll,pitch,yaw)
    map_dtype=np.dtype(map_dtype)
    if map_dtype not in (np.dtype(np.float32),np.dtype(np.float64)):
        raise ValueError("map_dtype must be float32 or float64")
    lx=np.empty((n,n),map_dtype); ly=np.empty((n,n),map_dtype)
    rx=np.empty((n,n),map_dtype); ry=np.empty((n,n),map_dtype)
    x=np.arange(n,dtype=np.float64)
    lon=x*(math.pi/n)-math.pi/2.0
    sx=np.sin(lon); cx=np.cos(lon)
    br=max(1,int(block_rows))
    for y0 in range(0,n,br):
        y1=min(n,y0+br); y=np.arange(y0,y1,dtype=np.float64)
        lat=math.pi/2.0-y*(math.pi/n); sl=np.sin(lat)[:,None]; cl=np.cos(lat)[:,None]
        bh=y1-y0
        d=np.empty((bh,n,3),np.float64)
        d[:,:,0]=cl*sx[None,:]; d[:,:,1]=sl; d[:,:,2]=cl*cx[None,:]
        flat=d.reshape(-1,3); dl=flat@Z.T; dr=dl@R
        px,py=_directions_to_source_coordinates((lh,lw),lp,dl)
        qx,qy=_directions_to_source_coordinates((rh,rw),rp,dr)
        lx[y0:y1]=px.reshape(bh,n); ly[y0:y1]=py.reshape(bh,n)
        rx[y0:y1]=qx.reshape(bh,n); ry[y0:y1]=qy.reshape(bh,n)
        if progress: progress(y1,n,"Preparing fixed video maps")
    return VideoRemapPlan(n,lx,ly,rx,ry)


def prepare_video_plan_for_sampling(plan,sampling="fast"):
    """Optimize coordinate maps for the selected video sampling mode.

    OpenCV's fixed-point remap representation is smaller and often faster for
    bilinear remapping.  Fast mode converts once, then releases four float maps
    to reduce steady-state memory.  HQ/native retain float maps.
    """
    s=str(sampling or "fast").lower()
    if s in ("fast","opencv","linear") and plan.left_map1 is None:
        if plan.left_x is None or plan.left_y is None or plan.right_x is None or plan.right_y is None:
            return plan
        lx=plan.left_x.astype(np.float32,copy=False); ly=plan.left_y.astype(np.float32,copy=False)
        rx=plan.right_x.astype(np.float32,copy=False); ry=plan.right_y.astype(np.float32,copy=False)
        plan.left_map1,plan.left_map2=cv2.convertMaps(lx,ly,cv2.CV_16SC2)
        plan.right_map1,plan.right_map2=cv2.convertMaps(rx,ry,cv2.CV_16SC2)
        plan.left_x=plan.left_y=plan.right_x=plan.right_y=None
    return plan


def render_video_frame_with_plan(left,right,plan,sampling="fast",block_rows=64,
                                 right_shift_x_deg=0.0,right_shift_y_deg=0.0,
                                 executor=None):
    n=int(plan.output_height); sampling=str(sampling or "fast").lower()
    if sampling in ("fast","opencv","linear","hq","high","lanczos","lanczos4"):
        interp=cv2.INTER_LANCZOS4 if sampling in ("hq","high","lanczos","lanczos4") else cv2.INTER_LINEAR
        if sampling in ("fast","opencv","linear") and plan.left_map1 is not None:
            lm=(plan.left_map1,plan.left_map2); rm=(plan.right_map1,plan.right_map2)
        else:
            lx=plan.left_x if plan.left_x.dtype==np.float32 else plan.left_x.astype(np.float32)
            ly=plan.left_y if plan.left_y.dtype==np.float32 else plan.left_y.astype(np.float32)
            rx=plan.right_x if plan.right_x.dtype==np.float32 else plan.right_x.astype(np.float32)
            ry=plan.right_y if plan.right_y.dtype==np.float32 else plan.right_y.astype(np.float32)
            lm=(lx,ly); rm=(rx,ry)
        def one(img,maps):
            return cv2.remap(img,maps[0],maps[1],interp,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        if executor is not None:
            fl=executor.submit(one,left,lm); fr=executor.submit(one,right,rm)
            L=fl.result(); R=fr.result()
        else:
            L=one(left,lm); R=one(right,rm)
        out=np.concatenate([L,R],axis=1)
        return apply_right_eye_trim_sbs(out,right_shift_x_deg,right_shift_y_deg,interp)
    if sampling not in ("native","exact"):
        raise ValueError("sampling must be 'fast', 'hq', or 'native'")
    out=np.zeros((n,2*n,3),np.uint8); br=max(1,int(block_rows))
    for y0 in range(0,n,br):
        y1=min(n,y0+br); valid=np.ones((y1-y0)*n,dtype=bool)
        out[y0:y1,:n]=bilinear_sample(
            left,plan.left_x[y0:y1].reshape(-1),plan.left_y[y0:y1].reshape(-1),valid
        ).reshape(y1-y0,n,3)
        out[y0:y1,n:]=bilinear_sample(
            right,plan.right_x[y0:y1].reshape(-1),plan.right_y[y0:y1].reshape(-1),valid
        ).reshape(y1-y0,n,3)
    return apply_right_eye_trim_sbs(out,right_shift_x_deg,right_shift_y_deg,cv2.INTER_LINEAR)


class _FfmpegPipeWriter:
    """Small cv::VideoWriter-like wrapper for portable HEVC output.

    180Kino 2.0 uses OpenCV cudacodec::VideoWriter with Codec::HEVC.  The
    clean-room clone uses FFmpeg when the user selects HEVC so that the same
    container/codec family is available without requiring an OpenCV CUDA build.
    """
    def __init__(self,path,width,height,fps,encoder="libx265"):
        ffmpeg=shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("HEVC output requires ffmpeg in PATH")
        self.path=os.fspath(path)
        self.width=int(width); self.height=int(height); self.fps=float(fps)
        pixels=self.width*self.height
        # The supplied native 8192x4096 sample is about 100 Mbit/s.  Scale the
        # target with pixel count for smaller diagnostic outputs.
        mbps=max(8,int(round(100.0*pixels/(8192.0*4096.0))))
        cmd=[ffmpeg,"-y","-hide_banner","-loglevel","error",
             "-f","rawvideo","-pix_fmt","bgr24",
             "-s:v",f"{self.width}x{self.height}",
             "-r",f"{self.fps:.12g}","-i","-","-an",
             "-c:v",encoder,"-pix_fmt","yuv420p"]
        if encoder=="hevc_nvenc":
            cmd += ["-preset","p5","-b:v",f"{mbps}M","-maxrate",f"{mbps}M","-bufsize",f"{mbps*2}M"]
        else:
            cmd += ["-preset","medium","-b:v",f"{mbps}M","-maxrate",f"{mbps}M","-bufsize",f"{mbps*2}M"]
        if Path(self.path).suffix.lower() in (".mp4",".m4v",".mov"):
            cmd += ["-tag:v","hvc1","-movflags","+faststart"]
        cmd += [self.path]
        self._cmd=cmd
        self._proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        self._released=False

    def write(self,frame):
        if self._released:
            raise RuntimeError("HEVC writer is already closed")
        if frame is None or frame.shape[:2] != (self.height,self.width):
            raise ValueError(f"Writer expected {self.width}x{self.height} BGR frame")
        if frame.dtype != np.uint8:
            frame=np.clip(frame,0,255).astype(np.uint8)
        if not frame.flags.c_contiguous:
            frame=np.ascontiguousarray(frame)
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            err=(self._proc.stderr.read() or b"").decode("utf-8","replace")
            raise RuntimeError("FFmpeg HEVC encoder stopped: "+err.strip())

    def release(self):
        if self._released:return
        self._released=True
        try:
            if self._proc.stdin:self._proc.stdin.close()
        except Exception:
            pass
        err=(self._proc.stderr.read() or b"").decode("utf-8","replace")
        rc=self._proc.wait()
        if rc:
            raise RuntimeError("FFmpeg HEVC encoder failed: "+err.strip())


class _FfmpegH264PreviewWriter:
    """Small H.264 proxy writer fed directly from already-rendered BGR frames.

    This avoids re-decoding a finished 4K/8K HEVC file just to create a browser
    preview.  The proxy is generated in parallel with the authoritative output,
    so the additional cost is only one downscale plus a lightweight H.264 encode
    per frame.
    """
    def __init__(self,path,width,height,fps):
        ffmpeg=shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found for browser preview encoding")
        self.path=os.fspath(path);self.width=int(width);self.height=int(height);self.fps=float(fps)
        cmd=[ffmpeg,"-y","-hide_banner","-loglevel","error",
             "-f","rawvideo","-pix_fmt","bgr24","-s:v",f"{self.width}x{self.height}",
             "-r",f"{self.fps:.12g}","-i","-","-an","-c:v","libx264",
             "-preset","ultrafast","-crf","24","-pix_fmt","yuv420p",
             "-movflags","+faststart",self.path]
        self._proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        self._released=False

    def write(self,frame):
        if self._released: raise RuntimeError("Preview writer is already closed")
        if frame.shape[:2]!=(self.height,self.width):
            raise ValueError(f"Preview writer expected {self.width}x{self.height}")
        if not frame.flags.c_contiguous: frame=np.ascontiguousarray(frame)
        try:self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            err=(self._proc.stderr.read() or b"").decode("utf-8","replace")
            raise RuntimeError("H.264 preview encoder stopped: "+err.strip())

    def release(self):
        if self._released:return
        self._released=True
        try:
            if self._proc.stdin:self._proc.stdin.close()
        except Exception:pass
        err=(self._proc.stderr.read() or b"").decode("utf-8","replace")
        rc=self._proc.wait()
        if rc: raise RuntimeError("H.264 preview encoder failed: "+err.strip())


def _round_away_from_zero(x):
    x=float(x)
    return math.floor(x+0.5) if x>=0 else math.ceil(x-0.5)


def _open_video_writer(path,width,height,fps,codec="auto"):
    ext=Path(path).suffix.lower()
    c=str(codec or "auto").strip().lower()
    if c in ("hevc","h265","native-hevc","hevc-software"):
        return _FfmpegPipeWriter(path,width,height,fps,"libx265"),"HEVC/libx265"
    if c in ("hevc-nvenc","nvenc","hevc_nvenc"):
        return _FfmpegPipeWriter(path,width,height,fps,"hevc_nvenc"),"HEVC/NVENC"
    candidates=[]
    raw=str(codec or "auto").strip()
    if raw and raw.lower()!="auto": candidates.append(raw[:4])
    candidates += (["avc1","H264","mp4v"] if ext in (".mp4",".m4v",".mov") else ["MJPG","XVID","mp4v"])
    tried=[]
    for cs in candidates:
        if len(cs)!=4 or cs in tried: continue
        tried.append(cs)
        wr=cv2.VideoWriter(os.fspath(path),cv2.VideoWriter_fourcc(*cs),float(fps),(int(width),int(height)))
        if wr.isOpened(): return wr,cs
        wr.release()
    raise RuntimeError("Could not open video writer; tried "+", ".join(tried)+
                       ". Select HEVC (requires ffmpeg), AVI/MJPG, or install an OpenCV/FFmpeg build with MP4 encoding support.")

def convert_stereo_video(left_path,right_path,output_path,lp,rp,
                         left_a,left_b,right_a,right_b,
                         left_start,left_end,right_start,right_end,
                         output_width=4096,roll=0.0,pitch=0.0,yaw=0.0,
                         codec="auto",fps=None,fps_mode="kino",sampling="fast",
                         right_shift_x_deg=0.0,right_shift_y_deg=0.0,
                         parallel_eyes=None,map_block_rows=64,sample_block_rows=64,
                         preview_path=None,preview_width=1280,
                         length_policy="strict",
                         progress=None,stop_requested=None):
    """Convert synchronized frame ranges to SBS VR180.

    ``length_policy`` controls what happens when the selected ranges contain a
    different number of frames:

    * ``strict``      -- reject unequal ranges (old behaviour).
    * ``trim``        -- render only the shorter number of pairs.
    * ``repeat_last`` -- extend the shorter side by holding its last real frame.
    * ``black``       -- extend the shorter side with generated black frames.

    Holding the last frame is usually the least distracting way to cope with a
    camera that stopped a little early.  It does *not* synchronize two cameras;
    synchronization is still established by choosing the correct L/R start or
    reference-frame offset first.

    Frame ranges are zero-based and inclusive.  Audio is intentionally omitted,
    matching the documented 180Kino 2.0 behaviour.
    """
    ls,le,rs,re=map(int,(left_start,left_end,right_start,right_end))
    lc=le-ls+1; rc=re-rs+1
    if lc<=0 or rc<=0: raise ValueError("End frame must be >= start frame")
    policy=str(length_policy or "strict").strip().lower()
    aliases={"hold":"repeat_last","repeat":"repeat_last","pad":"repeat_last","pad_last":"repeat_last","shorter":"trim"}
    policy=aliases.get(policy,policy)
    if policy not in ("strict","trim","repeat_last","black"):
        raise ValueError("length_policy must be strict, trim, repeat_last, or black")
    if policy=="strict" and lc!=rc:
        raise ValueError(f"Left/right ranges must contain the same number of frames (left {lc}, right {rc}); choose trim or an extend policy to continue")
    total=min(lc,rc) if policy=="trim" else max(lc,rc)
    if policy=="strict": total=lc

    output_width=int(output_width)
    if output_width<128 or output_width%2: raise ValueError("Output width must be an even integer >= 128")
    n=output_width//2
    capL=cv2.VideoCapture(os.fspath(left_path)); capR=cv2.VideoCapture(os.fspath(right_path))
    if not capL.isOpened() or not capR.isOpened():
        capL.release();capR.release();raise IOError("Could not open one or both input videos")
    writer=None;preview_writer=None;preview_error=None;preview_w=preview_h=0
    try:
        capL.set(cv2.CAP_PROP_POS_FRAMES,ls); capR.set(cv2.CAP_PROP_POS_FRAMES,rs)
        okL,frameL=capL.read(); okR,frameR=capR.read()
        if not okL or frameL is None or not okR or frameR is None:
            raise RuntimeError("Could not read first synchronized frame pair")
        # Keep immutable templates for generated padding.  Repeat-last simply
        # leaves frameL/frameR untouched after the selected range ends.
        blackL=np.zeros_like(frameL); blackR=np.zeros_like(frameR)
        source_fps=float(capL.get(cv2.CAP_PROP_FPS) or 0.0) or float(capR.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        if fps is None or float(fps)<=0:
            fps=source_fps
        fps_mode=str(fps_mode or "kino").strip().lower()
        if fps_mode in ("kino","180kino","native","rounded"):
            fps=float(_round_away_from_zero(fps))
        elif fps_mode not in ("source","preserve","exact"):
            raise ValueError("fps_mode must be 'kino' or 'source'")
        def mp(done,count,stage):
            if progress: progress(0,total,stage,0.10*(done/max(1,count)))
        map_dtype=np.float64 if str(sampling).lower() in ("native","exact") else np.float32
        plan=build_video_remap_plan(frameL.shape,frameR.shape,lp,rp,left_a,left_b,right_a,right_b,
                                    n,roll,pitch,yaw,map_block_rows,mp,map_dtype)
        plan=prepare_video_plan_for_sampling(plan,sampling)
        Path(output_path).parent.mkdir(parents=True,exist_ok=True)
        writer,used_codec=_open_video_writer(output_path,output_width,n,fps,codec)
        if preview_path:
            try:
                preview_w=min(max(320,int(preview_width)),output_width);preview_w-=preview_w%2
                preview_h=max(2,int(round(preview_w*n/output_width)));preview_h-=preview_h%2
                Path(preview_path).parent.mkdir(parents=True,exist_ok=True)
                preview_writer=_FfmpegH264PreviewWriter(preview_path,preview_w,preview_h,fps)
            except Exception as e:
                preview_error=str(e);preview_writer=None
        if parallel_eyes is None:
            try: parallel_eyes=(cv2.getNumThreads()<=1)
            except Exception: parallel_eyes=False
        ex=ThreadPoolExecutor(max_workers=2) if parallel_eyes and str(sampling).lower() not in ("native","exact") else None
        try:
            for i in range(total):
                if stop_requested and stop_requested(): raise InterruptedError("Video conversion cancelled")
                if i:
                    # Advance only while a real selected frame still exists on
                    # that side.  Once exhausted, apply the requested padding.
                    if i < lc:
                        okL,nextL=capL.read()
                        if not okL or nextL is None: raise RuntimeError(f"Input ended at L{ls+i}")
                        frameL=nextL
                    elif policy=="black":
                        frameL=blackL
                    # repeat_last => deliberately retain the previous frameL
                    if i < rc:
                        okR,nextR=capR.read()
                        if not okR or nextR is None: raise RuntimeError(f"Input ended at R{rs+i}")
                        frameR=nextR
                    elif policy=="black":
                        frameR=blackR
                out=render_video_frame_with_plan(
                    frameL,frameR,plan,sampling,sample_block_rows,
                    right_shift_x_deg,right_shift_y_deg,ex
                )
                writer.write(out)
                if preview_writer is not None:
                    try:
                        pv=cv2.resize(out,(preview_w,preview_h),interpolation=cv2.INTER_AREA)
                        preview_writer.write(pv)
                    except Exception as e:
                        preview_error=str(e)
                        try:preview_writer.release()
                        except Exception:pass
                        preview_writer=None
                ltag=(f"L{ls+i}" if i<lc else ("L hold" if policy=="repeat_last" else "L black"))
                rtag=(f"R{rs+i}" if i<rc else ("R hold" if policy=="repeat_last" else "R black"))
                if progress: progress(i+1,total,f"Rendering {ltag} / {rtag}",0.10+0.90*((i+1)/total))
        finally:
            if ex is not None: ex.shutdown(wait=True)
        writer.release(); writer=None
        if preview_writer is not None:
            try:preview_writer.release()
            except Exception as e:preview_error=str(e)
            preview_writer=None
        return {"output":os.fspath(output_path),"width":output_width,"height":n,
                "fps":float(fps),"source_fps":float(source_fps),"fps_mode":fps_mode,
                "frames":total,"left_frames":lc,"right_frames":rc,"length_policy":policy,
                "codec":used_codec,"audio":False,"sampling":sampling,
                "right_shift_x_deg":float(right_shift_x_deg),"right_shift_y_deg":float(right_shift_y_deg),
                "parallel_eyes":bool(parallel_eyes),
                "preview_path":os.fspath(preview_path) if preview_path and Path(preview_path).exists() else None,
                "preview_width":int(preview_w),"preview_height":int(preview_h),"preview_error":preview_error}
    finally:
        if writer is not None: writer.release()
        if preview_writer is not None:
            try:preview_writer.release()
            except Exception:pass
        capL.release(); capR.release()

def print_profile(p):
    print(f'Name: {p.name}\nMagnification: {p.magnification}\nRadius mode: {p.radius_mode}\nRadius: {p.radius}\nProjection mode: {p.projection.mode}\nCustom formula: {p.projection.custom_formula}\nOptical axis mode: {p.optical_axis_mode}\nOptical axis: {p.optical_axis_x}, {p.optical_axis_y}')
    if p.projection.mode==3:
        print('Custom formula:',p.projection.custom_formula)
        print('Breakpoints:',p.projection.breakpoints)
        for i,c in enumerate(p.projection.coeffs):print(f'  segment {i}: A={c[0]:.15g}, B={c[1]:.15g}, C={c[2]:.15g}')

def cmd_profiles(a):
    for i,p in enumerate(parse_profiles(a.ini)):print(f'[{i}] {p.name} | radius_mode={p.radius_mode} radius={p.radius:g} projection={p.projection.mode} axis=({p.optical_axis_x:g},{p.optical_axis_y:g}) axis_mode={p.optical_axis_mode}')
def cmd_info(a):print_profile(profile_by_name(parse_profiles(a.ini),a.profile))
def cmd_test(a):
    p=profile_by_name(parse_profiles(a.ini),a.profile); print_profile(p); print('\nProjection test')
    for t in np.linspace(0,math.pi/2,11):
        y=projection_forward(t,p.projection); inv=projection_inverse_scalar(y,p.projection); print(f'theta={t:.9f} r/R={y:.9f} inv={inv:.9f} err={inv-t:+.3e}')
    print('\nGeometry test'); R=calculate_radius(p,a.width,a.height); cx,cy=calculate_optical_axis(p,a.width,a.height); print('radius=',R,'center=',cx,cy)
    for x,y in [(cx,cy),(cx+R*.25,cy),(cx+R*.5,cy),(cx,cy+R*.25),(cx-R*.4,cy+R*.3)]:
        ray=pixel_to_ray(x,y,p,a.width,a.height); print(f'pixel=({x:.2f},{y:.2f}) ray={ray} back={ray_to_pixel(ray,p,a.width,a.height)}')
    print('\nStereo sanity'); aa=normalize(np.array([.15,.1,.98])); bb=normalize(np.array([-.2,.3,.93])); K=Rz(math.radians(17))@Ry(math.radians(-11)); rr1=K.T@aa; rr2=K.T@bb; rec=stereo_rotation(aa,bb,rr1,rr2); print('A error',np.linalg.norm(rec@rr1-aa)); print('B error',np.linalg.norm(rec@rr2-bb)); print(rec)
def cmd_synth(a):
    p=profile_by_name(parse_profiles(a.ini),a.profile); write_image(a.output,make_synthetic_fisheye(p,a.width,a.height)); print('Wrote',a.output)
def cmd_match(a):
    l=read_image(a.left); r=read_image(a.right); rx,ry,s=template_match_point(l,r,a.x,a.y,a.template); print(f'left=({a.x},{a.y}) right=({rx:.3f},{ry:.3f}) sqdiff={s:.9f}')
def cmd_render(a):
    ps=parse_profiles(a.ini); lp=profile_by_name(ps,a.left_profile); rp=profile_by_name(ps,a.right_profile); l=read_image(a.left); r=read_image(a.right); 
    if l is None:raise FileNotFoundError(a.left)
    if r is None:raise FileNotFoundError(a.right)
    o=render_stereo_exe_geometry(l,r,lp,rp,(a.left_ax,a.left_ay),(a.left_bx,a.left_by),(a.right_ax,a.right_ay),(a.right_bx,a.right_by),a.height,a.roll,a.pitch,a.yaw); write_image(a.output,o); print(f'Wrote {a.output} ({o.shape[1]}x{o.shape[0]})')
def cmd_gopro(a):
    p = gopro12_max_lens_mod2_profile(
        a.width,
        a.height,
        a.fov,
    )
    print_profile(p)
    print("\nDerived values")
    print("--------------")
    print(f"Image size:          {a.width} x {a.height}")
    print(f"Assumed FOV:         {a.fov} degrees")
    print(f"Radius:              {p.radius}")
    print(f"Optical axis:        ({p.optical_axis_x}, {p.optical_axis_y})")
    print(f"theta_edge:          {math.radians(a.fov/2):.12f} rad")
    print(f"r/R model:           theta / theta_edge")
    print(
        f"equidistant scale:   "
        f"{1/math.radians(a.fov/2):.12f}"
    )
    print(
        "\nWARNING: this is an approximate test profile. "
        "Real Max Lens Mod 2.0 output is digitally processed and "
        "must be calibrated from actual frames for accurate VR180."
    )


def cmd_stereo_exe(a):
    ps=parse_profiles(a.ini)
    lp=profile_by_name(ps,a.left_profile); rp=profile_by_name(ps,a.right_profile)
    la=pixel_to_ray(a.lax,a.lay,lp,a.lw,a.lh); lb=pixel_to_ray(a.lbx,a.lby,lp,a.lw,a.lh)
    ra=pixel_to_ray(a.rax,a.ray,rp,a.rw,a.rh); rb=pixel_to_ray(a.rbx,a.rby,rp,a.rw,a.rh)
    R=stereo_rotation(la,lb,ra,rb)
    print("R (right -> left):\n",R)
    print("A error:",np.linalg.norm(R@ra-la))
    print("B error:",np.linalg.norm(R@rb-lb))
    print("det(R):",np.linalg.det(R))
    print("orthogonality:",np.linalg.norm(R.T@R-np.eye(3)))


def cmd_render_exe(a):
    ps=parse_profiles(a.ini); lp=profile_by_name(ps,a.left_profile); rp=profile_by_name(ps,a.right_profile); l=read_image(a.left); r=read_image(a.right)
    if l is None: raise FileNotFoundError(a.left)
    if r is None: raise FileNotFoundError(a.right)
    o=render_stereo_exe_geometry(l,r,lp,rp,(a.left_ax,a.left_ay),(a.left_bx,a.left_by),(a.right_ax,a.right_ay),(a.right_bx,a.right_by),a.height,a.roll,a.pitch,a.yaw)
    write_image(a.output,o); print(f'Wrote {a.output} ({o.shape[1]}x{o.shape[0]})')


def cmd_projection(a):
    p=profile_by_name(parse_profiles(a.ini),a.profile); print('theta(rad),theta(deg),r/R')
    for t in np.linspace(0,math.pi/2,a.samples):print(f'{t:.12f},{math.degrees(t):.8f},{projection_forward(t,p.projection):.12f}')

# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------


def _resolve_cli_profile(name,side,width,height,profiles=None):
    profiles=profiles or embedded_profiles()
    if name==GOPRO_CAL_L_NAME:return gopro12_max_lens_mod2_calibrated_profile("L",width,height)
    if name==GOPRO_CAL_R_NAME:return gopro12_max_lens_mod2_calibrated_profile("R",width,height)
    if name.startswith("GoPro HERO12 Black + Max Lens Mod 2.0"):
        return gopro12_max_lens_mod2_profile(width,height,177.0)
    return profile_by_name(profiles,name)


def cmd_video(a):
    li=video_probe(a.left); ri=video_probe(a.right)
    ps=parse_profiles(a.ini) if a.ini else embedded_profiles()
    lp=_resolve_cli_profile(a.left_profile,"L",li["width"],li["height"],ps)
    rp=_resolve_cli_profile(a.right_profile,"R",ri["width"],ri["height"],ps)
    def cb(done,total,stage,frac):
        print(f"\r{stage}: {frac*100:5.1f}%",end="",flush=True)
    result=convert_stereo_video(
        a.left,a.right,a.output,lp,rp,
        (a.left_ax,a.left_ay),(a.left_bx,a.left_by),
        (a.right_ax,a.right_ay),(a.right_bx,a.right_by),
        a.left_start,a.left_end,a.right_start,a.right_end,
        output_width=a.width,roll=a.roll,pitch=a.pitch,yaw=a.yaw,
        codec=a.codec,fps=a.fps,fps_mode=a.fps_mode,sampling=a.sampling,length_policy=a.length_policy,progress=cb)
    print("\n"+json.dumps(result,indent=2))


def cmd_clip_video(a):
    def cb(done,total,stage):
        print(f"\r{stage}: {done}/{total}",end="",flush=True)
    files=clip_video_jpegs(a.left,a.right,a.output_dir,
                           a.left_start,a.left_end,a.right_start,a.right_end,
                           a.jpeg_quality,cb)
    print(f"\nWrote {len(files)} JPEGs to {a.output_dir}")


# ---------------------------------------------------------------------------
# Human-readable mechanics / pipeline documentation used by the desktop Info
# window.  Keeping this explanation next to the implementation helps prevent
# the UI documentation from becoming disconnected from the actual algorithms.
# ---------------------------------------------------------------------------
INFO_SECTIONS = [
    ("Overview", """
180pyugen converts a synchronized pair of fisheye views into a VR180
side-by-side equirectangular image or movie.  Each output eye is a square that
covers 180 degrees horizontally and 180 degrees vertically; the left and right
squares are then packed next to each other.

The program does not merely slide the two source pictures until they look
similar.  It models each clicked/matched image point as a 3-D viewing ray.  Two
correspondence pairs, A and B, constrain a 3-D rotation between the two camera
coordinate systems.  That rotation is applied to the right eye during
rendering, which is why distant scenery can remain well aligned even when the
two physical cameras were not mounted with perfectly identical orientation.

The desktop and web/server versions share the same Python geometry engine.
The web application performs preview and interaction in the browser but sends
final rendering and native-style matching to Python/OpenCV on the server.
"""),
    ("Still-image pipeline", """
STAGE 1 — Decode input
The left and right images are decoded into BGR/RGB raster arrays.  OpenCV is
the default decoder because it is closest to the original applications.  A
Pillow decoder remains available for diagnostic comparisons.

STAGE 2 — Resolve lens profile
For each eye the selected profile defines the 180-degree image radius, optical
axis coordinates, and fisheye mapping function.  Radius can be explicit or
computed from image geometry.  The optical axis may be the image center or an
explicit calibrated coordinate.

STAGE 3 — Convert A/B pixels to unit rays
A source pixel is expressed relative to the optical axis.  Its radial distance
is inverted through the lens projection function to recover polar angle theta;
its azimuth comes from the pixel direction around the optical axis.  The result
is a normalized 3-D camera ray.

STAGE 4 — Stereo rotation
The right A ray is first rotated onto the left A ray using the shortest
Rodrigues rotation.  A second rotation around the already-aligned A axis brings
right B onto left B.  This two-stage construction is the recovered native-style
stereo alignment used by the project.

STAGE 5 — Generate VR180 output rays
For every pixel of each square output eye, the renderer creates the
corresponding direction in a 180x180-degree equirectangular field.  Optional
roll, pitch and yaw are applied as a 3-D zenith correction.

STAGE 6 — Project back into the fisheye image
Each direction is mapped through the selected lens model into source-image X/Y
coordinates.  The source is sampled with either the reconstructed native-style
bilinear path or an optimized OpenCV interpolation path.

STAGE 7 — Pack and encode
The left and right square images are placed side by side, producing the normal
2:1 VR180 stereo frame.  PNG/JPEG are supported for still output.
"""),
    ("Video pipeline", """
VIDEO INPUT AND SYNCHRONIZATION
The two movies are independently probed for resolution, frame count, frame
rate and codec.  Frame numbers are zero-based.  A synchronized reference pair
is selected, for example L180 / R186 in the supplied 180Kino tutorial.

REFERENCE-FRAME DISPLAY
As soon as both movie files are selected, 180pyugen now opens the current
reference frame from each movie and displays them automatically.  The explicit
'Load reference frames' control is kept for changing frame numbers later.

A/B ACQUISITION
A and B are selected on the large left reference view.  The 180Kino-oriented
matcher uses a native 20x20 template with TM_SQDIFF_NORMED and chooses the
minimum score on the right reference frame.  Manual right-side correction is
still possible.

PLAN PRECOMPUTATION
The expensive geometry is independent of video content, so it is calculated
once: stereo rotation, zenith rotation, output direction field, source X/Y maps
and validity masks.  Reusing these maps is the main performance difference
between efficient video conversion and naively running the full still renderer
for every frame.

FRAME LOOP
For each synchronized frame pair, the precomputed maps are used to sample left
and right inputs.  'Fast' uses optimized OpenCV remapping; 'HQ' uses Lanczos4;
'Native' uses the slower reconstructed sampler for parity studies.  The eyes
are packed, optional final convergence/vertical trim is applied, and the frame
is sent to the video encoder.

ENCODING
HEVC/H.265 is the native-oriented choice.  NVENC can be used when available.
The 180Kino timing mode rounds the source rate in the same practical manner as
the native tutorial sample (59.94 -> 60 fps).  Audio is currently not muxed in
the compatibility path.
"""),
    ("Camera and lens mapping", """
The renderer separates camera geometry from stereo alignment.  A profile has
three major parts:

1. Image radius for 180 degrees.  This describes how far from the optical axis
   a 90-degree off-axis ray lands.
2. Optical axis X/Y.  Real fisheye lenses are frequently not perfectly centered
   in the raster, so calibrated coordinates can matter strongly at the edges.
3. Projection function.  Supported forms include orthographic, equisolid,
   arbitrary k in y=A*sin(k*theta), and the piecewise custom mode.

The 180Augen and 180Kino custom modes are intentionally distinguished.
180Kino's documented mode-3 coefficients describe a piecewise quadratic
function y=a*theta^2+b*theta+c.  180Augen's recovered executable path uses a
different custom expression, so merging the two formulas would reduce parity.

A good profile therefore controls lens distortion; A/B controls the relative
orientation of the two cameras.  They solve different problems and should not
be used interchangeably.
"""),
    ("Stereo Align / overlay tools", """
The Stereo Align inspector is a visual quality-control tool.  It can show a
normal alpha blend, absolute difference, red/cyan anaglyph, or a negative
invert-and-mix view.  In negative mode, well aligned structure tends toward a
flat middle gray while displaced edges remain visually pronounced.

The preview Right X/Y offsets are diagnostic source-image shifts.  They help a
human estimate residual displacement and inspect whether A/B were chosen well.
They intentionally do NOT rewrite the calibrated lens geometry.

For actual final-output correction, use Right-eye convergence X and vertical
trim Y.  Those controls operate in VR degrees on the rendered right eye.  They
are appropriate for small residual convergence or vertical-disparity cleanup,
but large values usually indicate that A/B or the camera profile should be
revisited instead.
"""),
    ("Quality and performance", """
FAST
Uses precomputed source maps and OpenCV remap.  Where possible the float maps
are converted to OpenCV's compact fixed-point representation.  This reduces
steady-state map memory and improves cache behavior.  It is the recommended
mode for ordinary video conversion.

HQ
Uses the same geometric maps but Lanczos4 interpolation.  It can preserve fine
high-frequency detail better when the source is being resampled substantially,
at the cost of more CPU time.

NATIVE
Uses the reconstructed native-style interpolation rules.  It is primarily a
comparison/debugging path because exact per-pixel arithmetic is much slower
than OpenCV's optimized remap implementation.

THREADING
OpenCV normally uses its own internal thread pool.  180pyugen therefore avoids
blindly running both eyes in extra Python threads when OpenCV is already
multithreaded; oversubscription can make rendering slower rather than faster.

VIDEO ENCODING
HEVC software encoding gives broad reproducibility.  HEVC NVENC moves encoding
to supported NVIDIA hardware.  Encoding acceleration does not accelerate the
fisheye remap itself unless a future GPU remap backend is enabled as well.
"""),
    ("Web/server architecture", """
The browser is responsible for ordinary interaction: file selection, canvas
preview, point picking, Stereo Align visualization, progress display and result
playback/download.

Python/FastAPI is responsible for authoritative processing: decoding server
inputs, native-style template matching, profile evaluation, geometry, frame
remapping and encoding.  Large files are stored in temporary per-session
directories.  Render work can run inline for development or through Redis/RQ
workers in a production deployment.

The web version automatically uploads and displays the selected video reference
frames once both movies are chosen.  Repeated renders reuse the uploaded
session rather than transferring the movie files again.
"""),
    ("Native parity and limitations", """
180pyugen is a clean-room reconstruction, not the original 180Augen/180Kino
source code.  Several details have been recovered directly from executable
behavior and validated against native outputs, including the two-point stereo
rotation strategy, template matching methods, lens profile structure and much
of the still-image sampling path.

The 180Kino application used CUDA and its own GPU conversion kernel.  The
portable 180pyugen video path currently reproduces the same conceptual camera
geometry but uses CPU/OpenCV remapping unless a selected codec uses GPU
encoding.  Native output videos and clipped frames are used as ground-truth
references while remaining pixel-level differences are investigated.

When exact parity and maximum speed pull in different directions, the program
keeps separate modes rather than hiding the trade-off: Native for comparison,
Fast for throughput, and HQ for resampling quality.
"""),
]


def show_information_page(parent):
    """Open an extensive, read-only explanation of the application's mechanics."""
    if tk is None:
        return
    win=tk.Toplevel(parent)
    win.title("180pyugen — Info")
    win.geometry("1080x780")
    win.minsize(760,520)
    outer=ttk.Frame(win,padding=8); outer.pack(fill="both",expand=True)
    ttk.Label(
        outer,
        text="180pyugen processing reference",
        font=("TkDefaultFont",14,"bold")
    ).pack(anchor="w",pady=(0,6))
    ttk.Label(
        outer,
        text="This page describes what the program does, where the stereo correction happens, and which quality/performance modes alter the pipeline.",
        wraplength=1000
    ).pack(anchor="w",pady=(0,8))
    nb=ttk.Notebook(outer); nb.pack(fill="both",expand=True)
    for title,body in INFO_SECTIONS:
        frame=ttk.Frame(nb,padding=8)
        nb.add(frame,text=title)
        txt=tk.Text(frame,wrap="word",undo=False,padx=12,pady=12)
        sb=ttk.Scrollbar(frame,orient="vertical",command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left",fill="both",expand=True)
        sb.pack(side="right",fill="y")
        txt.insert("1.0",textwrap.dedent(body).strip()+"\n")
        txt.configure(state="disabled")


def apply_tk_ui_theme(root, dark=True):
    """Apply a coherent light/dark ttk palette to an existing Tk hierarchy.

    ttk does not inherit Tk canvas colors from a theme, so callers still update
    their image canvases explicitly.  The helper intentionally sticks to the
    built-in ``clam`` theme to avoid external theme packages and to keep the
    desktop build portable across Linux/Windows installations.
    """
    style=ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    if dark:
        bg="#1b1d21"; panel="#24272d"; field="#17191d"; fg="#e7e9ed"; muted="#aeb4bf"; accent="#3b82f6"; border="#4b515c"
    else:
        bg="#f4f4f4"; panel="#ffffff"; field="#ffffff"; fg="#111111"; muted="#555555"; accent="#202020"; border="#b5b5b5"
    try: root.configure(bg=bg)
    except Exception: pass
    style.configure(".",background=bg,foreground=fg,fieldbackground=field,bordercolor=border,lightcolor=border,darkcolor=border)
    style.configure("TFrame",background=bg)
    style.configure("TLabelframe",background=bg,foreground=fg)
    style.configure("TLabelframe.Label",background=bg,foreground=fg)
    style.configure("TLabel",background=bg,foreground=fg)
    style.configure("TButton",background=panel,foreground=fg,bordercolor=border,padding=5)
    style.map("TButton",background=[("active",accent)],foreground=[("active","#ffffff")])
    style.configure("TEntry",fieldbackground=field,foreground=fg,insertcolor=fg)
    style.configure("TCombobox",fieldbackground=field,foreground=fg,background=panel,arrowcolor=fg)
    style.map("TCombobox",fieldbackground=[("readonly",field)],foreground=[("readonly",fg)])
    style.configure("TSpinbox",fieldbackground=field,foreground=fg,arrowcolor=fg)
    style.configure("Horizontal.TProgressbar",background=accent,troughcolor=field)
    return {"bg":bg,"panel":panel,"field":field,"fg":fg,"muted":muted,"accent":accent,"border":border,"canvas":"#111317" if dark else "#202020"}

class AugenGUI:
    """Small native Tkinter front-end for the reconstructed 180Augen pipeline.

    The UI intentionally follows the original application's workflow:
      * left/right image filenames
      * independent left/right camera profiles
      * two corresponding points A and B
      * roll/pitch/yaw correction
      * output equirectangular height
      * Render button and preview

    The geometry engine below remains the test/reconstruction implementation;
    the GUI does not claim that the unrecovered EXE details are exact.
    """
    def __init__(self, root, left_path=None, right_path=None):
        if tk is None or Image is None:
            raise RuntimeError("GUI requires tkinter and Pillow")
        self.root = root
        self.root.title("180pyugen v35 — Image + Video VR180")
        self.root.geometry("1550x900")
        self.root.minsize(1200, 760)

        self.profiles = embedded_profiles()
        self.gopro_profile = None
        self.left_img = None
        self.right_img = None
        self.left_tk = None
        self.right_tk = None
        self.left_scale = 1.0
        self.right_scale = 1.0
        self.left_offset = (0, 0)
        self.right_offset = (0, 0)
        self.overlay_window = None
        self.overlay_canvas = None
        self.overlay_tk = None
        self.overlay_alpha_var = tk.DoubleVar(value=0.50)
        self.overlay_mode_var = tk.StringVar(value="Perceptual fuse")
        self.overlay_dx_var = tk.DoubleVar(value=0.0)
        self.overlay_dy_var = tk.DoubleVar(value=0.0)
        # Stereo Align v30 works in FINAL projected-eye space.  These caches
        # hold a low-resolution render after the 3-D A/B transform and current
        # convergence/vertical-trim settings.  Candidate dx/dy therefore map
        # directly to the same right-eye translation used by the real output.
        self._overlay_eye_left = None
        self._overlay_eye_right = None
        self._overlay_auto_shift = (0.0,0.0,0.0)
        self.stereo_x_var = tk.DoubleVar(value=0.0)
        self.stereo_y_var = tk.DoubleVar(value=0.0)
        self.point_mode = None
        self.editing_label = None
        self.points = {"A": [(466.0,757.0),(455.0,767.0)], "B": [(1286.0,823.0),(1271.0,832.0)]}

        self.left_var = tk.StringVar(value=left_path or "")
        self.right_var = tk.StringVar(value=right_path or "")
        self.left_profile_var = tk.StringVar(value=GOPRO_CAL_L_NAME)
        self.right_profile_var = tk.StringVar(value=GOPRO_CAL_R_NAME)
        self.output_var = tk.StringVar(value="LROut.jpg")
        self.height_var = tk.IntVar(value=1024)
        self.roll_var = tk.DoubleVar(value=0.0)
        self.pitch_var = tk.DoubleVar(value=0.0)
        self.yaw_var = tk.DoubleVar(value=0.0)
        self.template_var = tk.IntVar(value=20)
        self.decoder_var = tk.StringVar(value="OpenCV")
        self.theme_var = tk.StringVar(value="Dark")
        self.ui_palette = apply_tk_ui_theme(self.root, True)
        self.status_var = tk.StringVar(value="Load left/right images, then set points A and B.")
        self.left_ax_var = tk.StringVar(value="466")
        self.left_ay_var = tk.StringVar(value="757")
        self.left_bx_var = tk.StringVar(value="1286")
        self.left_by_var = tk.StringVar(value="823")
        self.right_ax_var = tk.StringVar(value="455")
        self.right_ay_var = tk.StringVar(value="767")
        self.right_bx_var = tk.StringVar(value="1271")
        self.right_by_var = tk.StringVar(value="832")

        self._build()
        self._load_if_present()

    def _build(self):
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        top = ttk.Frame(root, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        top.columnconfigure(4, weight=1)

        ttk.Label(top, text="Left image:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.left_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(top, text="Open…", command=self._open_left).grid(row=0, column=2)
        ttk.Label(top, text="Right image:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.right_var).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(top, text="Open…", command=self._open_right).grid(row=1, column=2)
        ttk.Button(top, text="Load images", command=self.load_images).grid(row=0, column=3, rowspan=2, padx=10)
        ttk.Label(top,text="Theme").grid(row=0,column=4,padx=(12,3),sticky="e")
        ttk.Combobox(top,textvariable=self.theme_var,values=("Dark","Light"),state="readonly",width=8).grid(row=0,column=5,sticky="w")

        settings = ttk.LabelFrame(root, text="Camera / lens settings", padding=8)
        settings.grid(row=1, column=0, sticky="ew", padx=8, pady=(0,8))
        for c in range(4): settings.columnconfigure(c, weight=1)
        ttk.Label(settings, text="Left profile").grid(row=0,column=0,sticky="w")
        ttk.Label(settings, text="Right profile").grid(row=0,column=2,sticky="w")
        self.left_combo = ttk.Combobox(settings, textvariable=self.left_profile_var, state="readonly")
        self.right_combo = ttk.Combobox(settings, textvariable=self.right_profile_var, state="readonly")
        self.left_combo.grid(row=1,column=0,columnspan=2,sticky="ew",padx=(0,8))
        self.right_combo.grid(row=1,column=2,columnspan=2,sticky="ew")
        names = [p.name for p in self.profiles] + [GOPRO_CAL_L_NAME, GOPRO_CAL_R_NAME, "GoPro HERO12 Black + Max Lens Mod 2.0"]
        self.left_combo["values"] = names
        self.right_combo["values"] = names
        self.left_combo.bind("<<ComboboxSelected>>", lambda e: self._profile_changed("L"))
        self.right_combo.bind("<<ComboboxSelected>>", lambda e: self._profile_changed("R"))

        ttk.Label(settings,text="Output height").grid(row=2,column=0,sticky="w",pady=(8,0))
        ttk.Spinbox(settings, from_=256, to=8192, increment=256, textvariable=self.height_var, width=10).grid(row=3,column=0,sticky="w")
        ttk.Label(settings,text="Template size").grid(row=2,column=1,sticky="w",pady=(8,0))
        ttk.Spinbox(settings, from_=15, to=501, increment=2, textvariable=self.template_var, width=10).grid(row=3,column=1,sticky="w")
        ttk.Label(settings,text="Roll °").grid(row=2,column=2,sticky="w",pady=(8,0))
        ttk.Entry(settings,textvariable=self.roll_var,width=10).grid(row=3,column=2,sticky="w")
        ttk.Label(settings,text="Pitch °").grid(row=2,column=3,sticky="w",pady=(8,0))
        ttk.Entry(settings,textvariable=self.pitch_var,width=10).grid(row=3,column=3,sticky="w")
        ttk.Label(settings,text="Yaw °").grid(row=4,column=0,sticky="w",pady=(5,0))
        ttk.Entry(settings,textvariable=self.yaw_var,width=10).grid(row=5,column=0,sticky="w")

        ttk.Label(settings,text="Input decoder (test)").grid(row=4,column=1,sticky="w",pady=(5,0))
        self.decoder_combo=ttk.Combobox(
            settings, textvariable=self.decoder_var,
            values=("OpenCV","Pillow"), state="readonly", width=12
        )
        self.decoder_combo.grid(row=5,column=1,sticky="w")
        self.decoder_combo.bind("<<ComboboxSelected>>", self._decoder_changed)
        ttk.Label(settings,text="Right-eye convergence X °").grid(row=4,column=2,sticky="w",pady=(5,0))
        ttk.Entry(settings,textvariable=self.stereo_x_var,width=10).grid(row=5,column=2,sticky="w")
        ttk.Label(settings,text="Right-eye vertical trim Y °").grid(row=4,column=3,sticky="w",pady=(5,0))
        ttk.Entry(settings,textvariable=self.stereo_y_var,width=10).grid(row=5,column=3,sticky="w")

        body = ttk.Frame(root, padding=(8,0,8,8))
        self.preview_body=body
        body.grid(row=2,column=0,sticky="nsew")
        # Native 180Augen shows a large left picking image and a much smaller
        # right reference image.  Use roughly a 3:1 horizontal allocation.
        body.columnconfigure(0,weight=1,minsize=420)
        body.columnconfigure(1,weight=1,minsize=420)
        body.rowconfigure(0,weight=1)
        self.left_canvas = tk.Canvas(body, background="#202020", highlightthickness=1)
        self.right_canvas = tk.Canvas(body, background="#202020", highlightthickness=1)
        self.left_canvas.grid(row=0,column=0,sticky="nsew",padx=(0,6))
        self.right_canvas.grid(row=0,column=1,sticky="nsew",padx=(6,0))
        self.left_canvas.bind("<Button-1>", lambda e: self._canvas_click("L",e))
        self.right_canvas.bind("<Button-1>", lambda e: self._canvas_click("R",e))
        self.left_canvas.bind("<Configure>", lambda e: self._display("L"))
        self.right_canvas.bind("<Configure>", lambda e: self._display("R"))
        self.root.after_idle(self._apply_main_ui_options)
        self.theme_var.trace_add("write",lambda *_:self._apply_main_ui_options())

        bottom = ttk.Frame(root, padding=8)
        bottom.grid(row=3,column=0,sticky="ew")
        bottom.columnconfigure(1,weight=1)
        ttk.Button(bottom,text="Set A (click left)",command=lambda:self._start_point("A")).grid(row=0,column=0,padx=(0,5))
        ttk.Button(bottom,text="Set B (click left)",command=lambda:self._start_point("B")).grid(row=0,column=1,padx=5,sticky="w")
        ttk.Button(bottom,text="Get A, B automatically",command=self.auto_points).grid(row=0,column=2,padx=5)
        ttk.Button(bottom,text="Clear points",command=self.clear_points).grid(row=0,column=3,padx=5)
        ttk.Button(bottom,text="Overlay preview…",command=self.open_overlay_preview).grid(row=0,column=4,padx=5)
        ttk.Button(bottom,text="Render VR180",command=self.render).grid(row=0,column=5,padx=5)
        ttk.Button(bottom,text="Video workflow…",command=self.open_video_workflow).grid(row=0,column=6,padx=(12,5))
        ttk.Button(bottom,text="Info…",command=lambda:show_information_page(self.root)).grid(row=0,column=7,padx=(8,5))
        ttk.Label(bottom,text="Output:").grid(row=1,column=0,sticky="w",pady=(8,0))
        ttk.Entry(bottom,textvariable=self.output_var).grid(row=1,column=1,columnspan=3,sticky="ew",pady=(8,0),padx=5)
        ttk.Button(bottom,text="Save as…",command=self._save_as).grid(row=1,column=4,pady=(8,0))
        ttk.Label(bottom,textvariable=self.status_var,anchor="w").grid(row=2,column=0,columnspan=5,sticky="ew",pady=(8,0))

        pts = ttk.LabelFrame(bottom,text="Corresponding coordinates",padding=5)
        pts.grid(row=0,column=6,rowspan=3,padx=(15,0),sticky="ns")
        headers=["", "L x", "L y", "R x", "R y"]
        for j,h in enumerate(headers): ttk.Label(pts,text=h).grid(row=0,column=j,padx=3)
        vars_by_row=[("A",self.left_ax_var,self.left_ay_var,self.right_ax_var,self.right_ay_var),
                     ("B",self.left_bx_var,self.left_by_var,self.right_bx_var,self.right_by_var)]
        for i,row in enumerate(vars_by_row,1):
            ttk.Label(pts,text=row[0]).grid(row=i,column=0)
            for j,v in enumerate(row[1:],1): ttk.Entry(pts,textvariable=v,width=8).grid(row=i,column=j,padx=2)

    def _apply_main_ui_options(self):
        """Switch desktop palette and preview allocation without rebuilding UI."""
        dark=self.theme_var.get().lower().startswith("dark")
        self.ui_palette=apply_tk_ui_theme(self.root,dark)
        canvas_bg=self.ui_palette["canvas"]
        for c in (getattr(self,"left_canvas",None),getattr(self,"right_canvas",None)):
            if c is not None:
                try:c.configure(bg=canvas_bg,highlightbackground=self.ui_palette["border"])
                except Exception:pass
        body=getattr(self,"preview_body",None)
        if body is not None:
            body.columnconfigure(0,weight=1,minsize=420)
            body.columnconfigure(1,weight=1,minsize=420)
        self._display("L");self._display("R")

    def _load_if_present(self):
        if self.left_var.get() and self.right_var.get() and os.path.exists(self.left_var.get()) and os.path.exists(self.right_var.get()):
            self.load_images()

    def _open_left(self):
        p=filedialog.askopenfilename(filetypes=[("All files","*.*"),("All image files","*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),("JPEG","*.jpg *.jpeg"),("PNG","*.png"),("TIFF","*.tif *.tiff"),("BMP","*.bmp"),("WebP","*.webp")])
        if p:
            self.left_var.set(p)
            if self.right_var.get().strip(): self.load_images()
            else: self.status_var.set("Left image selected. Select the right image.")
    def _open_right(self):
        p=filedialog.askopenfilename(filetypes=[("All files","*.*"),("All image files","*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),("JPEG","*.jpg *.jpeg"),("PNG","*.png"),("TIFF","*.tif *.tiff"),("BMP","*.bmp"),("WebP","*.webp")])
        if p:
            self.right_var.set(p)
            if self.left_var.get().strip(): self.load_images()
            else: self.status_var.set("Right image selected. Select the left image.")
    def _save_as(self):
        p=filedialog.asksaveasfilename(
            initialfile=Path(self.output_var.get()).name or "LROut.jpg",
            defaultextension="",
            filetypes=[("All files","*.*"),("JPEG","*.jpg *.jpeg"),("PNG","*.png"),("TIFF","*.tif *.tiff"),("BMP","*.bmp"),("WebP","*.webp")]
        )
        if p:self.output_var.set(p)

    def _decoder_changed(self, event=None):
        if self.left_var.get().strip() and self.right_var.get().strip():
            self.load_images()

    def load_images(self):
        if cv2 is None:
            messagebox.showerror("OpenCV", "opencv-python is required")
            return
        dec=self.decoder_var.get()
        l=read_image(self.left_var.get(),dec); r=read_image(self.right_var.get(),dec)
        if l is None or r is None:
            messagebox.showerror("Image error", "Could not decode:\n\n" + "\n".join((["LEFT: "+self.left_var.get()] if l is None else []) + (["RIGHT: "+self.right_var.get()] if r is None else [])) + "\n\nSupported: JPG/JPEG, PNG, TIFF, BMP, WebP.")
            return
        self.left_img=cv2.cvtColor(l,cv2.COLOR_BGR2RGB); self.right_img=cv2.cvtColor(r,cv2.COLOR_BGR2RGB)

        gopro_test = (
            l.shape[1] == 1792 and l.shape[0] == 2048 and
            r.shape[1] == 1792 and r.shape[0] == 2048 and
            self.left_profile_var.get() == GOPRO_CAL_L_NAME and
            self.right_profile_var.get() == GOPRO_CAL_R_NAME
        )
        if gopro_test:
            self.points={
                "A":[(466.0,757.0),(455.0,767.0)],
                "B":[(1286.0,823.0),(1271.0,832.0)]
            }
            self._sync_vars()
        else:
            self.clear_points(redraw=False)

        self._display("L"); self._display("R"); self._refresh_overlay_preview()
        if gopro_test:
            self.status_var.set(
                f"Loaded GoPro calibration pair with {dec}. "
                "Preset A/B correspondences restored; ready to render."
            )
        else:
            self.status_var.set(f"Loaded with {dec}: L {l.shape[1]}×{l.shape[0]}, R {r.shape[1]}×{r.shape[0]}. Select A and B.")

    def _display(self, side):
        img=self.left_img if side=="L" else self.right_img
        canvas=self.left_canvas if side=="L" else self.right_canvas
        if img is None or canvas.winfo_width()<10 or canvas.winfo_height()<10:return
        h,w=img.shape[:2]; cw=max(10,canvas.winfo_width()-8); ch=max(10,canvas.winfo_height()-8)
        scale=min(cw/w,ch/h); nw=max(1,round(w*scale)); nh=max(1,round(h*scale))
        pil=Image.fromarray(img).resize((nw,nh),Image.Resampling.LANCZOS)
        tkimg=ImageTk.PhotoImage(pil); canvas.delete("all")
        ox=(canvas.winfo_width()-nw)//2; oy=(canvas.winfo_height()-nh)//2
        canvas.create_image(ox,oy,anchor="nw",image=tkimg)
        if side=="L": self.left_tk=tkimg; self.left_scale=scale; self.left_offset=(ox,oy)
        else: self.right_tk=tkimg; self.right_scale=scale; self.right_offset=(ox,oy)
        self._draw_points(side)

    def _draw_points(self,side):
        canvas=self.left_canvas if side=="L" else self.right_canvas; scale=self.left_scale if side=="L" else self.right_scale; ox,oy=self.left_offset if side=="L" else self.right_offset
        for label,(lp,rp) in self.points.items():
            pt=lp if side=="L" else rp
            if pt is None:continue
            x=ox+pt[0]*scale; y=oy+pt[1]*scale; z=7
            canvas.create_oval(x-z,y-z,x+z,y+z,outline="red",width=2)
            canvas.create_line(x-z*2,y,x+z*2,y,fill="red"); canvas.create_line(x,y-z*2,x,y+z*2,fill="red")
            canvas.create_text(x+12,y-12,text=label,fill="red",anchor="sw",font=("TkDefaultFont",11,"bold"))

    def _canvas_to_image(self,side,x,y):
        scale=self.left_scale if side=="L" else self.right_scale; ox,oy=self.left_offset if side=="L" else self.right_offset
        img=self.left_img if side=="L" else self.right_img
        if img is None or scale<=0:return None
        px=(x-ox)/scale; py=(y-oy)/scale
        h,w=img.shape[:2]
        if 0<=px<w and 0<=py<h:return float(px),float(py)
        return None

    def _canvas_click(self,side,event):
        pt=self._canvas_to_image(side,event.x,event.y)
        if pt is None:return
        if self.point_mode and side=="L":
            self._set_point(self.point_mode, "L", pt)
            self.status_var.set(f"{self.point_mode} left = ({pt[0]:.1f}, {pt[1]:.1f}); finding right correspondence…")
            if cv2 is not None and self.right_img is not None:
                bgr_l=cv2.cvtColor(self.left_img,cv2.COLOR_RGB2BGR); bgr_r=cv2.cvtColor(self.right_img,cv2.COLOR_RGB2BGR)
                try:
                    rx,ry,score=template_match_point(bgr_l,bgr_r,pt[0],pt[1],self.template_var.get())
                    self._set_point(self.point_mode,"R",(rx,ry)); self.status_var.set(f"{self.point_mode}: L ({pt[0]:.1f},{pt[1]:.1f}) → R ({rx:.1f},{ry:.1f}), score={score:.4f}")
                except Exception as e: self.status_var.set(f"Match failed: {e}")
            if getattr(self, "_two_point_mode", False) and self.point_mode == "A":
                self.point_mode = "B"
            else:
                self.point_mode=None
                self._two_point_mode=False
            self._display("L"); self._display("R"); self._refresh_overlay_preview()
        elif side=="R":
            # Manual correction: overwrite the most recently selected point.
            label=self.editing_label or self.point_mode or ("B" if self.points["A"][0] is not None else "A")
            self._set_point(label,"R",pt); self._display("R"); self._sync_vars(); self._refresh_overlay_preview()

    def _start_point(self,label):
        if self.left_img is None:
            messagebox.showwarning("No image","Load the left and right images first."); return
        self.point_mode=label; self.editing_label=label; self.status_var.set(f"Click the left image to set point {label}; the right point will be matched automatically.")

    def _set_point(self,label,side,pt):
        self.points[label][0 if side=="L" else 1]=pt; self._sync_vars()
    def _sync_vars(self):
        mapping=[("A",self.left_ax_var,self.left_ay_var,self.right_ax_var,self.right_ay_var),("B",self.left_bx_var,self.left_by_var,self.right_bx_var,self.right_by_var)]
        for lab,xv,yv,rxv,ryv in mapping:
            lp,rp=self.points[lab]
            xv.set("" if lp is None else f"{lp[0]:.3f}"); yv.set("" if lp is None else f"{lp[1]:.3f}")
            rxv.set("" if rp is None else f"{rp[0]:.3f}"); ryv.set("" if rp is None else f"{rp[1]:.3f}")
    def _read_vars(self):
        vals={}
        for lab,names in [("A",(self.left_ax_var,self.left_ay_var,self.right_ax_var,self.right_ay_var)),("B",(self.left_bx_var,self.left_by_var,self.right_bx_var,self.right_by_var))]:
            try: vals[lab]=[(float(v.get()),float(w.get())) for v,w in [(names[0],names[1]),(names[2],names[3])]]
            except ValueError: raise ValueError(f"Incomplete coordinates for point {lab}")
        return vals

    def clear_points(self,redraw=True):
        self.points={"A":[None,None],"B":[None,None]}; self._sync_vars(); self.point_mode=None; self.editing_label=None
        if redraw:self._display("L"); self._display("R")
        self._refresh_overlay_preview()
    def auto_points(self):
        """Choose two useful left-image points and immediately template-match them on right.

        This is a practical automatic replacement for the EXE's Get A,B workflow.
        The exact internal candidate-selection heuristic of 180Augen is not yet
        proven by disassembly, so we deliberately report the match scores.
        """
        if self.left_img is None or self.right_img is None:
            messagebox.showwarning("No images","Load both images first."); return
        if cv2 is None:
            messagebox.showerror("OpenCV","opencv-python is required."); return

        try:
            h,w=self.left_img.shape[:2]
            p=self._profile_for("L")
            cx,cy=calculate_optical_axis(p,w,h)
            rad=calculate_radius(p,w,h)

            # Candidate points stay well inside the fisheye and are separated.
            candidates=[]
            for sx,sy in [(-.42,-.22),(.42,.22),(-.34,.30),(.34,-.30),
                          (-.25,-.38),(.25,.38),(-.48,.05),(.48,-.05)]:
                x=cx+sx*rad; y=cy+sy*rad
                if rad > 0 and 0.12*rad <= math.hypot(x-cx,y-cy) <= .72*rad:
                    candidates.append((x,y))

            L=cv2.cvtColor(self.left_img,cv2.COLOR_RGB2BGR)
            R=cv2.cvtColor(self.right_img,cv2.COLOR_RGB2BGR)
            scored=[]
            for x,y in candidates:
                try:
                    rx,ry,score=template_match_point(L,R,x,y,int(self.template_var.get()))
                    scored.append((float(score),x,y,rx,ry))
                except Exception:
                    pass

            if len(scored) < 2:
                raise RuntimeError("Could not obtain two template matches.")

            scored.sort(key=lambda q:q[0])  # TM_SQDIFF: lower is better
            # Pick the strongest match, then the strongest sufficiently distant one.
            first=scored[0]
            min_sep=max(0.20*rad,80.0)
            second=None
            for q in scored[1:]:
                if math.hypot(q[1]-first[1],q[2]-first[2]) >= min_sep:
                    second=q
                    break
            if second is None:
                second=scored[1]

            chosen=[first,second]
            # Keep the conventional A/B order stable by left-image x.
            chosen.sort(key=lambda q:q[1])
            self.points={
                "A":[(chosen[0][1],chosen[0][2]),(chosen[0][3],chosen[0][4])],
                "B":[(chosen[1][1],chosen[1][2]),(chosen[1][3],chosen[1][4])]
            }
            self.point_mode=None
            self.editing_label=None
            self._sync_vars()
            self._display("L"); self._display("R")
            self.status_var.set(
                f"Automatic A/B complete: A SQDIFF={chosen[0][0]:.4f}, "
                f"B SQDIFF={chosen[1][0]:.4f}. Lower is better; adjust if needed."
            )
        except Exception as e:
            self.point_mode=None
            self.editing_label=None
            self.status_var.set(f"Automatic A/B failed: {e}")
            messagebox.showerror("Get A, B automatically",str(e))

    def _profile_for(self,side):
        name=(self.left_profile_var if side=="L" else self.right_profile_var).get()
        img=self.left_img if side=="L" else self.right_img
        if img is None: w,h=1792,2048
        else: h,w=img.shape[:2]

        if name == GOPRO_CAL_L_NAME:
            return gopro12_max_lens_mod2_calibrated_profile("L",w,h)
        if name == GOPRO_CAL_R_NAME:
            return gopro12_max_lens_mod2_calibrated_profile("R",w,h)
        if name == "GoPro HERO12 Black + Max Lens Mod 2.0":
            return gopro12_max_lens_mod2_profile(w,h,177.0)
        return profile_by_name(self.profiles,name)

    def _profile_changed(self,side):
        try:
            p=self._profile_for(side)
            self.status_var.set(f"{side} profile: radius={p.radius:.2f}, axis=({p.optical_axis_x:.2f},{p.optical_axis_y:.2f}), projection={p.projection.mode}, k={p.projection.k:.4f}")
        except Exception as e:self.status_var.set(str(e))

    def _prepare_projected_overlay(self):
        """Render a small copy of the ACTUAL final stereo-eye geometry.

        Earlier overlay versions compared the two raw fisheye source images.
        That was useful for seeing camera displacement, but a raw-pixel shift is
        not the shift applied to the final VR180 eye.  v30 instead renders the
        current calibrated/3-D-aligned stereo pair at 384 pixels per eye,
        including current convergence and vertical trim.  The candidate offset
        shown in the right-hand panel is therefore expressed in the final
        equirectangular eye coordinate system.
        """
        if self.left_img is None or self.right_img is None:
            raise ValueError("Load both images first")
        v=self._read_vars()
        lp=self._profile_for("L"); rp=self._profile_for("R")
        L=cv2.cvtColor(self.left_img,cv2.COLOR_RGB2BGR)
        R=cv2.cvtColor(self.right_img,cv2.COLOR_RGB2BGR)
        n=384
        out=render_stereo_exe_geometry(
            L,R,lp,rp,v["A"][0],v["B"][0],v["A"][1],v["B"][1],n,
            float(self.roll_var.get()),float(self.pitch_var.get()),float(self.yaw_var.get()))
        out=apply_right_eye_trim_sbs(out,float(self.stereo_x_var.get()),float(self.stereo_y_var.get()))
        rgb=cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
        self._overlay_eye_left=rgb[:,:n].copy(); self._overlay_eye_right=rgb[:,n:].copy()
        self._overlay_auto_shift=self._estimate_projected_overlay_shift(
            self._overlay_eye_left,self._overlay_eye_right)
        # The right half is explicitly an "aligned candidate", so initialize it
        # with the measured projected-eye residual rather than duplicating the
        # uncorrected current view.  Reset candidate still returns to 0/0.
        self.overlay_dx_var.set(self._overlay_auto_shift[0])
        self.overlay_dy_var.set(self._overlay_auto_shift[1])
        self._refresh_overlay_preview()

    @staticmethod
    def _estimate_projected_overlay_shift(left_rgb,right_rgb):
        """Estimate a global candidate translation in projected-eye pixels.

        Phase correlation is run on Laplacian edge images.  Stereo parallax is
        depth-dependent, so this is only an initial global suggestion; the
        manual drag/nudge remains authoritative.
        """
        L=cv2.cvtColor(left_rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
        R=cv2.cvtColor(right_rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
        L=cv2.Laplacian(L,cv2.CV_32F); R=cv2.Laplacian(R,cv2.CV_32F)
        h,w=L.shape
        win=cv2.createHanningWindow((w,h),cv2.CV_32F)
        (dx,dy),response=cv2.phaseCorrelate(R,L,win)
        limx=w*.25; limy=h*.25
        dx=float(np.clip(dx,-limx,limx)); dy=float(np.clip(dy,-limy,limy))
        return dx,dy,float(response)

    def _overlay_use_auto(self):
        dx,dy,response=self._overlay_auto_shift
        self.overlay_dx_var.set(dx); self.overlay_dy_var.set(dy)
        self.status_var.set(f"Projected auto-alignment candidate for fused preview: X {dx:.3f}px, Y {dy:.3f}px (response {response:.3f}).")
        self._refresh_overlay_preview()

    def _overlay_apply_to_output(self):
        """Commit the candidate preview shift to final stereo trim controls."""
        if self._overlay_eye_left is None:return
        n=float(self._overlay_eye_left.shape[1])
        dx=float(self.overlay_dx_var.get());dy=float(self.overlay_dy_var.get())
        self.stereo_x_var.set(float(self.stereo_x_var.get())+dx*180.0/n)
        self.stereo_y_var.set(float(self.stereo_y_var.get())+dy*180.0/n)
        self.status_var.set(
            f"Applied projected candidate to output: convergence {self.stereo_x_var.get():.4f}°, "
            f"vertical {self.stereo_y_var.get():.4f}°. The right panel now matches the real render setting.")
        self._prepare_projected_overlay()

    def _overlay_nudge(self,dx,dy):
        self.overlay_dx_var.set(float(self.overlay_dx_var.get())+float(dx))
        self.overlay_dy_var.set(float(self.overlay_dy_var.get())+float(dy))
        self._refresh_overlay_preview()

    def _overlay_drag_start(self,e):
        self._overlay_drag_state=(e.x,e.y,float(self.overlay_dx_var.get()),float(self.overlay_dy_var.get()))

    def _overlay_drag_motion(self,e):
        st=getattr(self,'_overlay_drag_state',None); sc=float(getattr(self,'_overlay_view_scale',1.0) or 1.0)
        if not st:return
        self.overlay_dx_var.set(st[2]+(e.x-st[0])/sc)
        self.overlay_dy_var.set(st[3]+(e.y-st[1])/sc)

    def open_overlay_preview(self):
        if self.left_img is None or self.right_img is None:
            messagebox.showwarning("Stereo Align", "Load both images first."); return
        if self.overlay_window is None or not self.overlay_window.winfo_exists():
            win=tk.Toplevel(self.root);win.title("Stereo Align — current vs aligned final-eye preview");win.geometry("1450x780");win.minsize(900,520)
            box=ttk.Frame(win,padding=8);box.pack(fill="both",expand=True)
            controls=ttk.Frame(box);controls.pack(fill="x")
            ttk.Label(controls,text="View").pack(side="left")
            ttk.Combobox(controls,textvariable=self.overlay_mode_var,values=("Perceptual fuse","Negative align","Alpha blend","Difference","Red/Cyan anaglyph"),state="readonly",width=18).pack(side="left",padx=5)
            ttk.Label(controls,text="Opacity").pack(side="left",padx=(8,0))
            ttk.Scale(controls,from_=0,to=1,variable=self.overlay_alpha_var,orient="horizontal",length=130,command=lambda *_:self._refresh_overlay_preview()).pack(side="left",padx=5)
            ttk.Label(controls,text="Candidate right-eye X/Y px").pack(side="left",padx=(8,0))
            ttk.Spinbox(controls,from_=-500,to=500,increment=.1,textvariable=self.overlay_dx_var,width=8).pack(side="left")
            ttk.Spinbox(controls,from_=-500,to=500,increment=.1,textvariable=self.overlay_dy_var,width=8).pack(side="left")
            ttk.Button(controls,text="Auto projected",command=self._overlay_use_auto).pack(side="left",padx=3)
            ttk.Button(controls,text="Apply to output",command=self._overlay_apply_to_output).pack(side="left",padx=3)
            ttk.Button(controls,text="Refresh projected view",command=self._prepare_projected_overlay).pack(side="left",padx=3)
            ttk.Button(controls,text="Reset candidate",command=lambda:(self.overlay_dx_var.set(0),self.overlay_dy_var.set(0),self._refresh_overlay_preview())).pack(side="left",padx=3)
            nudges=ttk.Frame(box);nudges.pack(fill="x",pady=(5,0))
            ttk.Label(nudges,text="Nudge candidate:").pack(side="left")
            for label,dx,dy in (("←",-1,0),("→",1,0),("↑",0,-1),("↓",0,1),("←0.1",-.1,0),("→0.1",.1,0),("↑0.1",0,-.1),("↓0.1",0,.1)):
                ttk.Button(nudges,text=label,command=lambda dx=dx,dy=dy:self._overlay_nudge(dx,dy),width=5).pack(side="left",padx=1)
            ttk.Label(nudges,text="LEFT = current fused final-view proxy. RIGHT = aligned candidate fused proxy. Apply to output writes the exact candidate into the final stereo render.").pack(side="left",padx=(10,0))
            self.overlay_canvas=tk.Canvas(box,bg="#111317",highlightthickness=1);self.overlay_canvas.pack(fill="both",expand=True,pady=(8,0))
            self.overlay_canvas.bind("<Configure>",lambda e:self._refresh_overlay_preview());self.overlay_canvas.bind("<ButtonPress-1>",self._overlay_drag_start);self.overlay_canvas.bind("<B1-Motion>",self._overlay_drag_motion)
            self.overlay_window=win
            self.overlay_mode_var.trace_add("write",lambda *_:self._refresh_overlay_preview());self.overlay_dx_var.trace_add("write",lambda *_:self._refresh_overlay_preview());self.overlay_dy_var.trace_add("write",lambda *_:self._refresh_overlay_preview())
        else:
            self.overlay_window.deiconify();self.overlay_window.lift()
        try:self._prepare_projected_overlay()
        except Exception as e:messagebox.showerror("Stereo Align",str(e),parent=self.overlay_window)

    def _refresh_overlay_preview(self):
        canvas=getattr(self,'overlay_canvas',None)
        if canvas is None or self.overlay_window is None or not self.overlay_window.winfo_exists():return
        L=self._overlay_eye_left;R=self._overlay_eye_right
        if L is None or R is None:
            canvas.delete("all")
            canvas.create_text(max(1,canvas.winfo_width())//2,max(1,canvas.winfo_height())//2,
                               text="Preparing projected alignment preview…",fill="#9aa6b2",
                               font=("TkDefaultFont",12,"bold"))
            return
        cw=max(1,canvas.winfo_width());ch=max(1,canvas.winfo_height());mode=self.overlay_mode_var.get()
        current=make_alignment_overlay_rgb(L,R,self.overlay_alpha_var.get(),mode,0,0)
        candidate=make_alignment_overlay_rgb(L,R,self.overlay_alpha_var.get(),mode,self.overlay_dx_var.get(),self.overlay_dy_var.get())
        h,w=current.shape[:2];gap=6
        combo=np.zeros((h,w*2+gap,3),dtype=np.uint8);combo[:,:w]=current;combo[:,w+gap:]=candidate
        sc=min((cw-8)/combo.shape[1],(ch-30)/h);self._overlay_view_scale=sc
        nw=max(1,round(combo.shape[1]*sc));nh=max(1,round(h*sc));ox=(cw-nw)//2;oy=max(24,(ch-nh)//2)
        im=Image.fromarray(combo).resize((nw,nh),Image.Resampling.LANCZOS);ph=ImageTk.PhotoImage(im)
        canvas.delete("all");canvas.create_image(ox,oy,anchor="nw",image=ph);self.overlay_tk=ph
        canvas.create_text(ox+round(w*sc)/2,8,text="CURRENT FUSED FINAL VIEW",fill="#d8dde6",anchor="n",font=("TkDefaultFont",11,"bold"))
        canvas.create_text(ox+round((w+gap+w/2)*sc),8,text="ALIGNED FUSED CANDIDATE",fill="#7ee7ff",anchor="n",font=("TkDefaultFont",11,"bold"))
        divx=ox+round((w+gap/2)*sc);canvas.create_line(divx,oy,divx,oy+nh,fill="#59616e",width=2)

    def render(self):
        try:
            if self.left_img is None or self.right_img is None: raise ValueError("Load both images first")
            v=self._read_vars()
            lp=self._profile_for("L"); rp=self._profile_for("R")
            L=cv2.cvtColor(self.left_img,cv2.COLOR_RGB2BGR); R=cv2.cvtColor(self.right_img,cv2.COLOR_RGB2BGR)
            la,lb=v["A"][0],v["B"][0]; ra,rb=v["A"][1],v["B"][1]
            # Use the EXE-oriented renderer in the GUI.  v7/v8 accidentally
            # kept calling the old conventional equirectangular renderer here,
            # so the UI preview/output did not exercise the reconstructed
            # 180Augen geometry at all.
            out=render_stereo_exe_geometry(
                L,R,lp,rp,la,lb,ra,rb,
                int(self.height_var.get()),
                float(self.roll_var.get()),
                float(self.pitch_var.get()),
                float(self.yaw_var.get())
            )
            out=apply_right_eye_trim_sbs(out,float(self.stereo_x_var.get()),float(self.stereo_y_var.get()))
            path=self.output_var.get() or "LROut.jpg"
            if not write_image(path,out): raise IOError(f"Could not write {path}")
            self.status_var.set(f"Rendered {path} — {out.shape[1]}×{out.shape[0]} — image workflow v30")
            self._show_preview(out)
        except Exception as e:
            messagebox.showerror("Render failed",str(e)); self.status_var.set(f"Render failed: {e}")

    def open_video_workflow(self):
        VideoWorkflow(self.root, self.profiles)

    def _show_preview(self,out):
        win=tk.Toplevel(self.root); win.title("VR180 preview")
        h,w=out.shape[:2]; maxw,maxh=1200,650; scale=min(maxw/w,maxh/h,1.0)
        rgb=cv2.cvtColor(out,cv2.COLOR_BGR2RGB); im=Image.fromarray(rgb).resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.Resampling.LANCZOS)
        ph=ImageTk.PhotoImage(im); lab=ttk.Label(win,image=ph); lab.image=ph; lab.pack(padx=8,pady=8)
        ttk.Label(win,text=f"{w} × {h}  |  {self.output_var.get()}").pack(pady=(0,8))



class VideoWorkflow:
    """180Kino-style workflow in a separate window inside 180pyugen."""

    def __init__(self,master,profiles):
        self.win=tk.Toplevel(master); self.win.title("180pyugen v35 — Video workflow (180Kino)")
        self.win.geometry("1500x900"); self.win.minsize(1150,720)
        self.profiles=list(profiles)
        self.left_frame=self.right_frame=None; self.left_tk=self.right_tk=None
        self.video_overlay_window=None; self.video_overlay_canvas=None; self.video_overlay_tk=None
        self.video_overlay_alpha=tk.DoubleVar(value=0.50)
        self.video_overlay_mode=tk.StringVar(value="Perceptual fuse")
        self.video_overlay_dx=tk.DoubleVar(value=0.0)
        self.video_overlay_dy=tk.DoubleVar(value=0.0)
        self._video_overlay_eye_left=None
        self._video_overlay_eye_right=None
        self._video_overlay_auto_shift=(0.0,0.0,0.0)
        # Reference-frame navigation state is independent for each eye and is
        # intentionally preserved when Previous/Next pair loads a new frame.
        # center is normalized source-image position; zoom=1 means fit-to-view.
        self.view_state={"L":{"zoom":1.0,"center":[0.5,0.5]},"R":{"zoom":1.0,"center":[0.5,0.5]}}
        self._pan_state=None
        self.transforms={"L":None,"R":None}; self.points={"A":[None,None],"B":[None,None]}; self.active=None
        self.worker=None; self.cancel_event=threading.Event()
        self.left_video=tk.StringVar(); self.right_video=tk.StringVar()
        self.left_ref=tk.IntVar(value=0); self.right_ref=tk.IntVar(value=0)
        self.left_start=tk.IntVar(value=0); self.left_end=tk.IntVar(value=0)
        self.right_start=tk.IntVar(value=0); self.right_end=tk.IntVar(value=0)
        self.left_profile=tk.StringVar(value=GOPRO_CAL_L_NAME); self.right_profile=tk.StringVar(value=GOPRO_CAL_R_NAME)
        self.output_width=tk.IntVar(value=4096); self.roll=tk.DoubleVar(value=0.0); self.pitch=tk.DoubleVar(value=0.0); self.yaw=tk.DoubleVar(value=0.0)
        self.output=tk.StringVar(value="VROut.mp4"); self.codec=tk.StringVar(value="auto"); self.sampling=tk.StringVar(value="fast"); self.fps_mode=tk.StringVar(value="kino")
        self.length_policy=tk.StringVar(value="strict")
        self.stereo_x=tk.DoubleVar(value=0.0); self.stereo_y=tk.DoubleVar(value=0.0)
        self.theme_var=tk.StringVar(value="Dark")
        self.last_output_path=None; self.output_check_btn=None
        self.ui_palette=apply_tk_ui_theme(self.win,True)
        self.status=tk.StringVar(value="Choose left/right movies and load synchronized reference frames.")
        self.progress=tk.DoubleVar(value=0.0)
        self.coord_vars={lab:[tk.StringVar() for _ in range(4)] for lab in ("A","B")}
        self._build()

    def _build(self):
        w=self.win; w.columnconfigure(0,weight=1); w.rowconfigure(3,weight=1)
        top=ttk.LabelFrame(w,text="Input movies / synchronization",padding=4); top.grid(row=0,column=0,sticky="ew",padx=6,pady=4)
        top.columnconfigure(1,weight=1)
        ttk.Label(top,text="Left movie").grid(row=0,column=0,sticky="w"); ttk.Entry(top,textvariable=self.left_video).grid(row=0,column=1,sticky="ew",padx=3)
        ttk.Button(top,text="Open…",command=lambda:self._pick_video("L")).grid(row=0,column=2)
        ttk.Label(top,text="Right movie").grid(row=1,column=0,sticky="w"); ttk.Entry(top,textvariable=self.right_video).grid(row=1,column=1,sticky="ew",padx=3)
        ttk.Button(top,text="Open…",command=lambda:self._pick_video("R")).grid(row=1,column=2)
        ttk.Label(top,text="Reference L").grid(row=0,column=3,padx=(15,4))
        self.left_ref_spin=ttk.Spinbox(top,from_=0,to=10**9,textvariable=self.left_ref,width=10)
        self.left_ref_spin.grid(row=0,column=4)
        ttk.Label(top,text="Reference R").grid(row=1,column=3,padx=(15,4))
        self.right_ref_spin=ttk.Spinbox(top,from_=0,to=10**9,textvariable=self.right_ref,width=10)
        self.right_ref_spin.grid(row=1,column=4)
        self.left_ref_spin.bind("<Return>",lambda e:self.load_reference_frames())
        self.right_ref_spin.bind("<Return>",lambda e:self.load_reference_frames())
        ttk.Button(top,text="Load refs",command=self.load_reference_frames).grid(row=0,column=5,rowspan=2,padx=3)
        ttk.Button(top,text="JPEG clip…",command=self.clip_dialog).grid(row=0,column=6,rowspan=2,padx=3)
        ttk.Button(top,text="Tutorial",command=self.tutorial_preset).grid(row=0,column=7,rowspan=2,padx=3)
        ttk.Button(top,text="Info…",command=lambda:show_information_page(self.win)).grid(row=0,column=8,rowspan=2,padx=3)
        ttk.Button(top,text="◀ pair",command=lambda:self._step_reference(-1)).grid(row=0,column=9,padx=2)
        ttk.Button(top,text="pair ▶",command=lambda:self._step_reference(1)).grid(row=1,column=9,padx=2)
        ttk.Label(top,text="Theme").grid(row=0,column=10,padx=(5,2)); ttk.Combobox(top,textvariable=self.theme_var,values=("Dark","Light"),state="readonly",width=8).grid(row=0,column=11)
        ttk.Button(top,text="Fit",command=self._fit_reference_views).grid(row=1,column=10,padx=(5,2))
        ttk.Button(top,text="1:1",command=self._one_to_one_views).grid(row=1,column=11,padx=2)

        settings=ttk.LabelFrame(w,text="Conversion settings",padding=8); settings.grid(row=1,column=0,sticky="ew",padx=8,pady=(0,8))
        for c in range(9): settings.columnconfigure(c,weight=1)
        names=[p.name for p in self.profiles]+[GOPRO_CAL_L_NAME,GOPRO_CAL_R_NAME,"GoPro HERO12 Black + Max Lens Mod 2.0"]
        ttk.Label(settings,text="Left profile").grid(row=0,column=0,sticky="w"); ttk.Combobox(settings,textvariable=self.left_profile,values=names,state="readonly").grid(row=1,column=0,columnspan=2,sticky="ew",padx=(0,8))
        ttk.Label(settings,text="Right profile").grid(row=0,column=2,sticky="w"); ttk.Combobox(settings,textvariable=self.right_profile,values=names,state="readonly").grid(row=1,column=2,columnspan=2,sticky="ew",padx=(0,8))
        ttk.Button(settings,text="GoPro profiles",command=self.gopro_video_profiles).grid(row=2,column=0,columnspan=2,sticky="w",pady=(4,0))
        ttk.Label(settings,text="Output width").grid(row=0,column=4,sticky="w"); ttk.Combobox(settings,textvariable=self.output_width,values=(2048,4096,6144,8192),width=10).grid(row=1,column=4,sticky="w")
        ttk.Label(settings,text="Sampling").grid(row=0,column=5,sticky="w"); ttk.Combobox(settings,textvariable=self.sampling,values=("fast","hq","native"),state="readonly",width=10).grid(row=1,column=5,sticky="w")
        ttk.Label(settings,text="Codec").grid(row=0,column=6,sticky="w"); ttk.Combobox(settings,textvariable=self.codec,values=("auto","hevc","hevc-nvenc","avc1","mp4v","MJPG"),width=12).grid(row=1,column=6,sticky="w")
        ttk.Label(settings,text="Roll / Pitch / Yaw").grid(row=0,column=7,sticky="w"); ang=ttk.Frame(settings); ang.grid(row=1,column=7,sticky="w")
        for j,v in enumerate((self.roll,self.pitch,self.yaw)): ttk.Entry(ang,textvariable=v,width=6).grid(row=0,column=j,padx=1)
        ttk.Label(settings,text="Timing").grid(row=0,column=8,sticky="w"); ttk.Combobox(settings,textvariable=self.fps_mode,values=("kino","source"),state="readonly",width=10).grid(row=1,column=8,sticky="w")
        ttk.Label(settings,text="Right-eye convergence X °").grid(row=2,column=4,sticky="w",pady=(5,0)); ttk.Entry(settings,textvariable=self.stereo_x,width=9).grid(row=3,column=4,sticky="w")
        ttk.Label(settings,text="Right-eye vertical trim Y °").grid(row=2,column=5,sticky="w",pady=(5,0)); ttk.Entry(settings,textvariable=self.stereo_y,width=9).grid(row=3,column=5,sticky="w")
        ttk.Label(settings,text="Performance").grid(row=2,column=6,sticky="w",pady=(5,0)); ttk.Label(settings,text="Fast/HQ remap left+right in parallel; HEVC-NVENC uses GPU encode.").grid(row=3,column=6,columnspan=3,sticky="w")

        ranges=ttk.LabelFrame(w,text="Frame ranges (zero-based, inclusive)",padding=8); ranges.grid(row=2,column=0,sticky="ew",padx=8,pady=(0,8))
        for col,(lab,var) in enumerate((("Left start",self.left_start),("Left end",self.left_end),("Right start",self.right_start),("Right end",self.right_end))):
            ttk.Label(ranges,text=lab).grid(row=0,column=2*col); ttk.Entry(ranges,textvariable=var,width=10).grid(row=0,column=2*col+1)
        ttk.Button(ranges,text="Trim to shorter remaining",command=self.use_full_ranges).grid(row=0,column=8,padx=(12,3))
        ttk.Button(ranges,text="Use full remaining + extend shorter",command=self.use_extended_ranges).grid(row=0,column=9,padx=3)
        ttk.Label(ranges,text="Length handling").grid(row=0,column=10,padx=(12,2))
        ttk.Combobox(ranges,textvariable=self.length_policy,
                     values=("strict","trim","repeat_last","black"),state="readonly",width=12).grid(row=0,column=11)

        body=ttk.Frame(w,padding=(8,0,8,8)); self.preview_body=body; body.grid(row=3,column=0,sticky="nsew")
        body.columnconfigure(0,weight=1,minsize=480); body.columnconfigure(1,weight=1,minsize=480); body.rowconfigure(0,weight=0)
        self.lc=tk.Canvas(body,bg="#202020",highlightthickness=1,height=500,cursor="fleur"); self.rc=tk.Canvas(body,bg="#202020",highlightthickness=1,height=500,cursor="fleur")
        self.lc.grid(row=0,column=0,sticky="ew",padx=(0,6)); self.rc.grid(row=0,column=1,sticky="ew",padx=(6,0))
        for side,canvas in (("L",self.lc),("R",self.rc)):
            canvas.bind("<ButtonPress-1>",lambda e,s=side:self._reference_press(s,e))
            canvas.bind("<B1-Motion>",lambda e,s=side:self._reference_drag(s,e))
            canvas.bind("<ButtonRelease-1>",lambda e,s=side:self._reference_release(s,e))
            canvas.bind("<MouseWheel>",lambda e,s=side:self._reference_wheel(s,e))
            canvas.bind("<Button-4>",lambda e,s=side:self._reference_wheel(s,e,1))
            canvas.bind("<Button-5>",lambda e,s=side:self._reference_wheel(s,e,-1))
            canvas.bind("<Configure>",lambda e,s=side:self._display(s))
        self.theme_var.trace_add("write",lambda *_:self._apply_video_ui_options()); self.win.after_idle(self._apply_video_ui_options)

        bottom=ttk.Frame(w,padding=4); bottom.grid(row=4,column=0,sticky="ew"); bottom.columnconfigure(5,weight=1)
        ttk.Button(bottom,text="Set A",command=lambda:self._set_active("A")).grid(row=0,column=0,padx=3); ttk.Button(bottom,text="Set B",command=lambda:self._set_active("B")).grid(row=0,column=1,padx=3)
        ttk.Button(bottom,text="Clear A/B",command=self.clear_points).grid(row=0,column=2,padx=3)
        ttk.Button(bottom,text="Overlay preview…",command=self.open_overlay_preview).grid(row=0,column=9,padx=(12,3))
        pts=ttk.LabelFrame(bottom,text="A/B coordinates",padding=2); pts.grid(row=0,column=3,rowspan=4,padx=8)
        for j,h in enumerate(("","L x","L y","R x","R y")): ttk.Label(pts,text=h).grid(row=0,column=j,padx=2)
        for i,lab in enumerate(("A","B"),1):
            ttk.Label(pts,text=lab).grid(row=i,column=0)
            for j,v in enumerate(self.coord_vars[lab],1): ttk.Entry(pts,textvariable=v,width=8).grid(row=i,column=j,padx=1)
        ttk.Label(bottom,text="Output").grid(row=0,column=4,sticky="e"); ttk.Entry(bottom,textvariable=self.output).grid(row=0,column=5,sticky="ew",padx=5)
        ttk.Button(bottom,text="Save as…",command=self._save_as).grid(row=0,column=6,padx=3)
        self.start_btn=ttk.Button(bottom,text="Start video conversion",command=self.start_conversion); self.start_btn.grid(row=0,column=7,padx=(10,3))
        self.cancel_btn=ttk.Button(bottom,text="Cancel",command=self.cancel_conversion,state="disabled"); self.cancel_btn.grid(row=0,column=8,padx=3)
        self.output_check_btn=ttk.Button(bottom,text="Inspect generated video…",command=self.open_output_frame_checker,state="disabled"); self.output_check_btn.grid(row=0,column=9,padx=(10,3))
        ttk.Progressbar(bottom,variable=self.progress,maximum=100).grid(row=1,column=4,columnspan=5,sticky="ew",pady=(8,0))
        ttk.Label(bottom,textvariable=self.status,anchor="w").grid(row=2,column=4,columnspan=5,sticky="ew",pady=(6,0))
        ttk.Label(bottom,text="Audio is not written in this 180Kino-compatible path.").grid(row=3,column=4,columnspan=5,sticky="w",pady=(4,0))

    def _apply_video_ui_options(self):
        """Apply the selected palette and keep both reference views equally sized."""
        dark=self.theme_var.get().lower().startswith("dark")
        self.ui_palette=apply_tk_ui_theme(self.win,dark)
        for c in (getattr(self,"lc",None),getattr(self,"rc",None)):
            if c is not None:
                try:c.configure(bg=self.ui_palette["canvas"],highlightbackground=self.ui_palette["border"])
                except Exception:pass
        body=getattr(self,"preview_body",None)
        if body is not None:
            body.columnconfigure(0,weight=1,minsize=480)
            body.columnconfigure(1,weight=1,minsize=480)
        self._display("L");self._display("R")

    def _fit_reference_views(self):
        """Return both reference canvases to fit-to-window without changing frames."""
        for s in ("L","R"):
            self.view_state[s]["zoom"]=1.0
            self.view_state[s]["center"]=[0.5,0.5]
            self._display(s)

    def _one_to_one_views(self):
        """Show source pixels at approximately 1 display pixel per source pixel.

        This is intentionally a *zoomed crop* for 4K sources.  The normalized
        center is preserved, so switching to Previous/Next pair keeps looking
        at the same timer/LED/detail rather than jumping back to image center.
        """
        for s in ("L","R"):
            frame=self.left_frame if s=="L" else self.right_frame
            canvas=self.lc if s=="L" else self.rc
            if frame is None: continue
            h,w=frame.shape[:2];cw=max(1,canvas.winfo_width());ch=max(1,canvas.winfo_height())
            fit=min(cw/w,ch/h)
            self.view_state[s]["zoom"]=max(1.0,1.0/max(fit,1e-9))
            self._display(s)

    @staticmethod
    def _clamp_reference_center(cx,cy,w,h,scale,cw,ch):
        # Clamp the source-space centre just enough to keep the image covering
        # the canvas at zoom > fit.  At fit scale the natural 0.5 centre wins.
        if w*scale<=cw: cx=w*0.5
        else:
            half=cw/(2*scale); cx=min(w-half,max(half,cx))
        if h*scale<=ch: cy=h*0.5
        else:
            half=ch/(2*scale); cy=min(h-half,max(half,cy))
        return cx,cy

    def _reference_press(self,side,e):
        # Point picking is explicitly armed by Set A / Set B and is one-shot.
        # Normal left-button interaction is therefore always safe for panning.
        if self.active in ("A","B"):
            self._click(side,e)
            self.active=None
            self._set_reference_cursor(False)
            self.status.set(self.status.get()+"  Navigation mode restored.")
            return
        st=self.view_state[side]
        self._pan_state=(side,e.x,e.y,float(st["center"][0]),float(st["center"][1]))

    def _reference_drag(self,side,e):
        st=self._pan_state
        tr=self.transforms.get(side)
        if not st or st[0]!=side or not tr:return
        scale,ox,oy,w,h=tr; dx=e.x-st[1];dy=e.y-st[2]
        cx=st[3]*w-dx/max(scale,1e-9);cy=st[4]*h-dy/max(scale,1e-9)
        cw=max(1,(self.lc if side=="L" else self.rc).winfo_width());ch=max(1,(self.lc if side=="L" else self.rc).winfo_height())
        cx,cy=self._clamp_reference_center(cx,cy,w,h,scale,cw,ch)
        self.view_state[side]["center"]=[cx/w,cy/h]
        self._display(side)

    def _reference_release(self,side,e):
        if self._pan_state and self._pan_state[0]==side:self._pan_state=None

    def _reference_wheel(self,side,e,direction=None):
        frame=self.left_frame if side=="L" else self.right_frame
        if frame is None:return "break"
        step=direction if direction is not None else (1 if getattr(e,"delta",0)>0 else -1)
        st=self.view_state[side];old=float(st["zoom"]);new=max(1.0,min(32.0,old*(1.20 if step>0 else 1/1.20)))
        st["zoom"]=new;self._display(side)
        return "break"

    def _step_reference(self,delta):
        """Move both synchronized reference indices together and redraw frames."""
        try:
            self.left_ref.set(max(0,int(self.left_ref.get())+int(delta)))
            self.right_ref.set(max(0,int(self.right_ref.get())+int(delta)))
            if self.left_video.get().strip() and self.right_video.get().strip():
                self.load_reference_frames()
        except Exception as e:
            self.status.set(str(e))

    def gopro_video_profiles(self):
        """Select GoPro lens profiles without overwriting capture-specific A/B.

        Earlier web builds incorrectly copied the A/B coordinates from the
        supplied GoPro *still calibration test scene* into every GoPro video.
        Those points describe scene/camera orientation for that one pair; they
        are not camera constants.  Reusing them can rotate the right eye by a
        completely wrong amount and make video output look unrelated to the
        otherwise-good still conversion.
        """
        self.left_profile.set(GOPRO_CAL_L_NAME)
        self.right_profile.set(GOPRO_CAL_R_NAME)
        msg="GoPro calibrated L/R profiles selected. Existing A/B preserved; choose fresh A/B from this video's reference frames."
        frame=self.left_frame if self.left_frame is not None else self.right_frame
        if frame is not None:
            ok,warn=gopro_calibration_compatibility(frame.shape[1],frame.shape[0])
            msg += "  " + warn
        self.status.set(msg)
        self._refresh_overlay_preview()

    def tutorial_preset(self):
        """Load the settings shown in the supplied 180Kino 2.0 tutorial."""
        self.left_ref.set(180)
        self.right_ref.set(186)
        self.left_start.set(180)
        self.left_end.set(480)
        self.right_start.set(186)
        self.right_end.set(486)
        self.left_profile.set("DJI Action2")
        self.right_profile.set("DJI Action2")
        self.output_width.set(8192)
        self.codec.set("hevc")
        self.fps_mode.set("kino")
        self.sampling.set("fast"); self.length_policy.set("strict")
        self.stereo_x.set(0.0); self.stereo_y.set(0.0)
        self.output.set("VROut.mp4")
        self.points={
            "A":[(1152.0,1272.0),(1202.0,1293.0)],
            "B":[(3180.0,1492.0),(3227.0,1522.0)],
        }
        self._sync_coords()
        self._display("L"); self._display("R"); self._refresh_overlay_preview()
        self.status.set(
            "180Kino tutorial preset: refs L180/R186, range L180..480 / "
            "R186..486, DJI Action2, tutorial A/B loaded."
        )
        if self.left_video.get() and self.right_video.get():
            try:
                self.load_reference_frames()
                # load_reference_frames clears points, so restore the tutorial values.
                self.points={
                    "A":[(1152.0,1272.0),(1202.0,1293.0)],
                    "B":[(3180.0,1492.0),(3227.0,1522.0)],
                }
                self._sync_coords()
                self._display("L"); self._display("R"); self._refresh_overlay_preview()
            except Exception:
                pass

    def _pick_video(self,side):
        # File selection and reference display are intentionally coupled.  In
        # earlier builds the files were accepted but the large/small reference
        # canvases stayed blank until the user separately pressed "Load
        # reference frames".  That was easy to interpret as a decode failure.
        p=filedialog.askopenfilename(filetypes=[("All files","*.*"),("Video files","*.mp4 *.mov *.m4v *.avi *.mkv *.webm")])
        if not p:
            return
        (self.left_video if side=="L" else self.right_video).set(p)
        try:
            info=video_probe(p)
            if side=="L":
                self.left_end.set(max(0,info["frame_count"]-1))
            else:
                self.right_end.set(max(0,info["frame_count"]-1))
            self.status.set(f"{side}: {info['width']}×{info['height']}, {info['frame_count']} frames, {info['fps']:.3f} fps, {info['fourcc'] or 'codec ?'}")

            # Once both sides exist, schedule the decode after Tk has finished
            # updating the file-entry widgets.  Using after_idle keeps the file
            # dialog callback short and lets the canvases obtain valid sizes
            # before _display() computes its fit scale.
            if self.left_video.get().strip() and self.right_video.get().strip():
                self.win.after_idle(self.load_reference_frames)
        except Exception as e:
            messagebox.showerror("Video",str(e),parent=self.win)

    def use_full_ranges(self):
        """Trim both ranges to the shorter remaining movie.

        This does NOT add frames.  It is the old "Use synchronized remaining
        length" behaviour, renamed to make the truncation explicit.
        """
        try:
            li=video_probe(self.left_video.get()); ri=video_probe(self.right_video.get())
            ls=max(0,int(self.left_start.get())); rs=max(0,int(self.right_start.get()))
            n=min(li["frame_count"]-ls,ri["frame_count"]-rs)
            if n<=0: raise ValueError("Start frame is outside one of the movies")
            self.left_end.set(ls+n-1);self.right_end.set(rs+n-1);self.length_policy.set("trim")
            self.status.set(f"Trimmed to shorter remaining length: {n} frame pairs (L{ls}..{ls+n-1}, R{rs}..{rs+n-1}).")
        except Exception as e: messagebox.showerror("Video",str(e),parent=self.win)

    def use_extended_ranges(self):
        """Keep every remaining real frame and extend the shorter eye."""
        try:
            li=video_probe(self.left_video.get());ri=video_probe(self.right_video.get())
            ls=max(0,int(self.left_start.get()));rs=max(0,int(self.right_start.get()))
            if ls>=li["frame_count"] or rs>=ri["frame_count"]:raise ValueError("Start frame is outside one of the movies")
            self.left_end.set(li["frame_count"]-1);self.right_end.set(ri["frame_count"]-1)
            self.length_policy.set("repeat_last")
            lc=li["frame_count"]-ls;rc=ri["frame_count"]-rs
            self.status.set(f"Full remaining ranges selected: left {lc}, right {rc}. Shorter side will hold its last frame for {abs(lc-rc)} frame(s).")
        except Exception as e:messagebox.showerror("Video",str(e),parent=self.win)

    def load_reference_frames(self):
        try:
            self.left_frame=read_video_frame(self.left_video.get(),self.left_ref.get()); self.right_frame=read_video_frame(self.right_video.get(),self.right_ref.get())
            # Deliberately preserve A/B and zoom/pan while stepping through the
            # timeline.  This makes a zoomed timer/LED region usable for visual
            # synchronization without accidentally destroying calibration.
            self._display("L");self._display("R"); self._refresh_overlay_preview()
            msg=f"Loaded L{self.left_ref.get()} / R{self.right_ref.get()}. Drag to pan, wheel to zoom; Set A/B explicitly to edit coordinates."
            if self.left_profile.get()==GOPRO_CAL_L_NAME or self.right_profile.get()==GOPRO_CAL_R_NAME:
                okL,wL=gopro_calibration_compatibility(self.left_frame.shape[1],self.left_frame.shape[0])
                okR,wR=gopro_calibration_compatibility(self.right_frame.shape[1],self.right_frame.shape[0])
                if not (okL and okR): msg += "  WARNING: " + (wL if not okL else wR)
            self.status.set(msg)
        except Exception as e: messagebox.showerror("Reference frames",str(e),parent=self.win)

    def _display(self,side):
        """Draw a persistent zoom/pan reference view with in-frame metadata."""
        frame=self.left_frame if side=="L" else self.right_frame; canvas=self.lc if side=="L" else self.rc
        canvas.delete("all")
        if frame is None:return
        cw=max(1,canvas.winfo_width());ch=max(1,canvas.winfo_height());h,w=frame.shape[:2]
        fit=min(cw/w,ch/h);st=self.view_state[side];zoom=max(1.0,float(st.get("zoom",1.0)));sc=fit*zoom
        cx=float(st["center"][0])*w;cy=float(st["center"][1])*h
        cx,cy=self._clamp_reference_center(cx,cy,w,h,sc,cw,ch);st["center"]=[cx/w,cy/h]
        dw=max(1,round(w*sc));dh=max(1,round(h*sc));ox=cw/2-cx*sc;oy=ch/2-cy*sc
        # Crop to the visible source rectangle *before* creating a Tk image.
        # At 1:1 on a 4K source this avoids allocating/repainting an entire 4K
        # PhotoImage for every pan gesture; only the pixels visible in the
        # reference canvas are converted.  The coordinate transform still
        # describes the full source image, so A/B and mouse mapping stay exact.
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        sx0=max(0,int(math.floor((-ox)/sc))); sy0=max(0,int(math.floor((-oy)/sc)))
        sx1=min(w,int(math.ceil((cw-ox)/sc))); sy1=min(h,int(math.ceil((ch-oy)/sc)))
        sx1=max(sx0+1,sx1); sy1=max(sy0+1,sy1)
        crop=Image.fromarray(rgb).crop((sx0,sy0,sx1,sy1))
        cdx=ox+sx0*sc;cdy=oy+sy0*sc
        cdw=max(1,round((sx1-sx0)*sc));cdh=max(1,round((sy1-sy0)*sc))
        if crop.size!=(cdw,cdh):crop=crop.resize((cdw,cdh),Image.Resampling.LANCZOS)
        ph=ImageTk.PhotoImage(crop)
        if side=="L":self.left_tk=ph
        else:self.right_tk=ph
        canvas.create_image(cdx,cdy,image=ph,anchor="nw");self.transforms[side]=(sc,ox,oy,w,h)
        idx=0 if side=="L" else 1
        for lab,color in (("A","#ff3030"),("B","#00cfff")):
            p=self.points[lab][idx]
            if p:
                x=ox+p[0]*sc;y=oy+p[1]*sc;canvas.create_rectangle(x-5,y-5,x+5,y+5,outline=color,width=2);canvas.create_text(x+8,y-8,text=lab,fill=color,anchor="sw")
        # Reference metadata is intentionally rendered *inside* each image view
        # so screenshots retain the exact frame number and reference points.
        ref=int(self.left_ref.get() if side=="L" else self.right_ref.get())
        def fmt(p):return "—" if p is None else f"{p[0]:.0f},{p[1]:.0f}"
        a=self.points["A"][idx];b=self.points["B"][idx]
        line1=f"{side} frame {ref}   zoom {zoom*100:.0f}%"
        line2=f"Reference coordinates  A {fmt(a)}   B {fmt(b)}"
        # Keep reference information directly over the picture, without an
        # opaque panel.  A tiny shadow keeps white/cyan text readable over
        # both bright and dark footage while preserving the image underneath.
        canvas.create_text(13,13,text=line1,fill="#000000",anchor="nw",font=("TkDefaultFont",10,"bold"))
        canvas.create_text(12,12,text=line1,fill="#f5f8fb",anchor="nw",font=("TkDefaultFont",10,"bold"))
        canvas.create_text(13,30,text=line2,fill="#000000",anchor="nw",font=("TkDefaultFont",9))
        canvas.create_text(12,29,text=line2,fill="#9ee8ff",anchor="nw",font=("TkDefaultFont",9))

    def _point_from_event(self,side,e):
        tr=self.transforms.get(side)
        if not tr:return None
        sc,ox,oy,w,h=tr;x=(e.x-ox)/sc;y=(e.y-oy)/sc
        return (x,y) if 0<=x<w and 0<=y<h else None

    def _click(self,side,e):
        if self.active not in ("A","B"):return
        p=self._point_from_event(side,e)
        if p is None:return
        lab=self.active
        if side=="L":
            self.points[lab][0]=p
            try:
                rx,ry,score=template_match_point_kino(self.left_frame,self.right_frame,p[0],p[1],20);self.points[lab][1]=(rx,ry)
                self.status.set(f"{lab}: L({p[0]:.1f},{p[1]:.1f}) -> R({rx:.1f},{ry:.1f}), SQDIFF_NORMED={score:.6f}")
            except Exception as ex:self.status.set(f"{lab} match failed: {ex}")
        else:self.points[lab][1]=p
        self._sync_coords();self._display("L");self._display("R"); self._refresh_overlay_preview()

    def _set_reference_cursor(self, picking=False):
        """Use an arrow while A/B placement is armed; pan cursor otherwise."""
        cur="arrow" if picking else "fleur"
        for canvas in (getattr(self,"lc",None),getattr(self,"rc",None)):
            if canvas is not None:
                try: canvas.configure(cursor=cur)
                except Exception: pass

    def _set_active(self,lab):
        self.active=lab
        self._set_reference_cursor(True)
        self.status.set(f"Point {lab} armed for ONE click. Click either reference frame; afterwards the canvases return to navigation/pan mode.")
    def clear_points(self,redraw=True):
        self.points={"A":[None,None],"B":[None,None]};self._sync_coords()
        if redraw:self._display("L");self._display("R")
        self._refresh_overlay_preview()
    def _sync_coords(self):
        for lab in ("A","B"):
            L,R=self.points[lab]; vals=(None if L is None else L[0],None if L is None else L[1],None if R is None else R[0],None if R is None else R[1])
            for v,x in zip(self.coord_vars[lab],vals):v.set("" if x is None else f"{x:.3f}")
    def _read_coords(self):
        out={}
        for lab in ("A","B"):
            v=self.coord_vars[lab]
            try:out[lab]=[(float(v[0].get()),float(v[1].get())),(float(v[2].get()),float(v[3].get()))]
            except ValueError:raise ValueError(f"Incomplete {lab} coordinates")
        self.points=out;return out
    def _profile(self,side,w,h):
        name=(self.left_profile if side=="L" else self.right_profile).get()
        if name==GOPRO_CAL_L_NAME:return gopro12_max_lens_mod2_calibrated_profile("L",w,h)
        if name==GOPRO_CAL_R_NAME:return gopro12_max_lens_mod2_calibrated_profile("R",w,h)
        if name.startswith("GoPro HERO12 Black + Max Lens Mod 2.0"):return gopro12_max_lens_mod2_profile(w,h,177.0)
        return profile_by_name(self.profiles,name)
    def _save_as(self):
        p=filedialog.asksaveasfilename(initialfile=self.output.get() or "VROut.mp4",filetypes=[("All files","*.*"),("MP4","*.mp4"),("AVI","*.avi")])
        if p:self.output.set(p)
    def clip_dialog(self):
        if not self.left_video.get() or not self.right_video.get():
            messagebox.showwarning("JPEG clipping","Choose both videos first.",parent=self.win);return

        dlg=tk.Toplevel(self.win);dlg.title("JPEG clipping tool");dlg.transient(self.win);dlg.resizable(False,False)
        box=ttk.Frame(dlg,padding=12);box.grid(row=0,column=0,sticky="nsew")
        ls=tk.IntVar(value=int(self.left_start.get())); le=tk.IntVar(value=int(self.left_end.get()))
        rs=tk.IntVar(value=int(self.right_start.get())); re=tk.IntVar(value=int(self.right_end.get()))
        q=tk.IntVar(value=95); out=tk.StringVar(value=str(Path.cwd()/"cut_img"))
        ttk.Label(box,text="Movies into JPEG pictures",font=("TkDefaultFont",10,"bold")).grid(row=0,column=0,columnspan=6,sticky="w",pady=(0,8))
        ttk.Label(box,text="Left").grid(row=1,column=0,sticky="e");ttk.Label(box,text=Path(self.left_video.get()).name).grid(row=1,column=1,columnspan=2,sticky="w",padx=5)
        ttk.Label(box,text="Right").grid(row=1,column=3,sticky="e");ttk.Label(box,text=Path(self.right_video.get()).name).grid(row=1,column=4,columnspan=2,sticky="w",padx=5)
        for row,(lab,a,b) in enumerate((("Left Frame Number",ls,le),("Right Frame Number",rs,re)),2):
            ttk.Label(box,text=lab).grid(row=row,column=0,sticky="e",pady=3)
            ttk.Label(box,text="Start").grid(row=row,column=1);ttk.Entry(box,textvariable=a,width=10).grid(row=row,column=2)
            ttk.Label(box,text="End").grid(row=row,column=3);ttk.Entry(box,textvariable=b,width=10).grid(row=row,column=4)
        ttk.Label(box,text="JPEG quality").grid(row=4,column=0,sticky="e",pady=3);ttk.Spinbox(box,from_=1,to=100,textvariable=q,width=8).grid(row=4,column=1,sticky="w")
        ttk.Label(box,text="Output folder").grid(row=5,column=0,sticky="e",pady=3);ttk.Entry(box,textvariable=out,width=50).grid(row=5,column=1,columnspan=4,sticky="ew",padx=5)
        ttk.Button(box,text="Browse…",command=lambda: (lambda d: out.set(d) if d else None)(filedialog.askdirectory(parent=dlg,title="Choose cut_img output folder"))).grid(row=5,column=5)
        pvar=tk.DoubleVar(value=0);msg=tk.StringVar(value="Frame numbers are zero-based and inclusive; left/right clip counts may differ.")
        ttk.Progressbar(box,variable=pvar,maximum=100).grid(row=6,column=0,columnspan=6,sticky="ew",pady=(8,3));ttk.Label(box,textvariable=msg).grid(row=7,column=0,columnspan=6,sticky="w")

        def start_clip():
            try:
                vals=(int(ls.get()),int(le.get()),int(rs.get()),int(re.get()))
                if min(vals)<0 or vals[1]<vals[0] or vals[3]<vals[2]: raise ValueError("Invalid frame range")
                dest=out.get().strip();
                if not dest: raise ValueError("Choose an output folder")
            except Exception as e:
                messagebox.showerror("JPEG clipping",str(e),parent=dlg);return
            ok.configure(state="disabled")
            def run():
                try:
                    def cb(done,total,stage): dlg.after(0,lambda d=done,t=total,s=stage:(pvar.set(100*d/max(1,t)),msg.set(s)))
                    files=clip_video_jpegs(self.left_video.get(),self.right_video.get(),dest,*vals,int(q.get()),cb)
                    dlg.after(0,lambda:self.status.set(f"Finished clipping {len(files)} JPEGs to {dest}"))
                    dlg.after(0,lambda:msg.set(f"Finished clipping {len(files)} JPEGs."))
                    dlg.after(0,lambda:ok.configure(state="normal"))
                except Exception as e:
                    dlg.after(0,lambda e=e:messagebox.showerror("JPEG clipping",str(e),parent=dlg));dlg.after(0,lambda:ok.configure(state="normal"))
            threading.Thread(target=run,daemon=True).start()
        buttons=ttk.Frame(box);buttons.grid(row=8,column=0,columnspan=6,pady=(8,0))
        ok=ttk.Button(buttons,text="OK",command=start_clip);ok.pack(side="left",padx=5);ttk.Button(buttons,text="Close",command=dlg.destroy).pack(side="left",padx=5)

    def _prepare_projected_overlay(self):
        """Render the selected reference pair through the real final-eye pipeline."""
        if self.left_frame is None or self.right_frame is None:
            raise ValueError("Load synchronized reference frames first")
        pts=self._read_coords();hL,wL=self.left_frame.shape[:2];hR,wR=self.right_frame.shape[:2]
        lp=self._profile("L",wL,hL);rp=self._profile("R",wR,hR);n=384
        out=render_stereo_exe_geometry(
            self.left_frame,self.right_frame,lp,rp,pts["A"][0],pts["B"][0],pts["A"][1],pts["B"][1],n,
            float(self.roll.get()),float(self.pitch.get()),float(self.yaw.get()))
        out=apply_right_eye_trim_sbs(out,float(self.stereo_x.get()),float(self.stereo_y.get()))
        rgb=cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
        self._video_overlay_eye_left=rgb[:,:n].copy();self._video_overlay_eye_right=rgb[:,n:].copy()
        self._video_overlay_auto_shift=AugenGUI._estimate_projected_overlay_shift(self._video_overlay_eye_left,self._video_overlay_eye_right)
        # Start the candidate panel at the measured residual alignment.  This
        # keeps CURRENT on the left untouched and makes ALIGNED CANDIDATE on
        # the right immediately meaningful.
        self.video_overlay_dx.set(self._video_overlay_auto_shift[0])
        self.video_overlay_dy.set(self._video_overlay_auto_shift[1])
        self._refresh_overlay_preview()

    def _overlay_use_auto(self):
        dx,dy,response=self._video_overlay_auto_shift;self.video_overlay_dx.set(dx);self.video_overlay_dy.set(dy)
        self.status.set(f"Projected auto-alignment candidate for fused preview: X {dx:.3f}px, Y {dy:.3f}px (response {response:.3f}).")
        self._refresh_overlay_preview()

    def _overlay_apply_to_output(self):
        if self._video_overlay_eye_left is None:return
        n=float(self._video_overlay_eye_left.shape[1]);dx=float(self.video_overlay_dx.get());dy=float(self.video_overlay_dy.get())
        self.stereo_x.set(float(self.stereo_x.get())+dx*180.0/n);self.stereo_y.set(float(self.stereo_y.get())+dy*180.0/n)
        self.status.set(f"Applied candidate to video output: convergence {self.stereo_x.get():.4f}°, vertical {self.stereo_y.get():.4f}°.")
        self._prepare_projected_overlay()

    def _overlay_nudge(self,dx,dy):
        self.video_overlay_dx.set(float(self.video_overlay_dx.get())+float(dx));self.video_overlay_dy.set(float(self.video_overlay_dy.get())+float(dy));self._refresh_overlay_preview()
    def _overlay_drag_start(self,e):self._overlay_drag_state=(e.x,e.y,float(self.video_overlay_dx.get()),float(self.video_overlay_dy.get()))
    def _overlay_drag_motion(self,e):
        st=getattr(self,'_overlay_drag_state',None);sc=float(getattr(self,'_overlay_view_scale',1.0) or 1.0)
        if not st:return
        self.video_overlay_dx.set(st[2]+(e.x-st[0])/sc);self.video_overlay_dy.set(st[3]+(e.y-st[1])/sc)

    def open_overlay_preview(self):
        if self.left_frame is None or self.right_frame is None:
            messagebox.showwarning("Stereo Align","Load synchronized reference frames first.",parent=self.win);return
        if self.video_overlay_window is None or not self.video_overlay_window.winfo_exists():
            win=tk.Toplevel(self.win);win.title("Video Stereo Align — current vs aligned final-eye preview");win.geometry("1450x780");win.minsize(900,520)
            box=ttk.Frame(win,padding=8);box.pack(fill="both",expand=True);controls=ttk.Frame(box);controls.pack(fill="x")
            ttk.Label(controls,text="View").pack(side="left");ttk.Combobox(controls,textvariable=self.video_overlay_mode,values=("Perceptual fuse","Negative align","Alpha blend","Difference","Red/Cyan anaglyph"),state="readonly",width=18).pack(side="left",padx=5)
            ttk.Label(controls,text="Opacity").pack(side="left",padx=(8,0));ttk.Scale(controls,from_=0,to=1,variable=self.video_overlay_alpha,orient="horizontal",length=130,command=lambda *_:self._refresh_overlay_preview()).pack(side="left",padx=5)
            ttk.Label(controls,text="Candidate right-eye X/Y px").pack(side="left",padx=(8,0));ttk.Spinbox(controls,from_=-500,to=500,increment=.1,textvariable=self.video_overlay_dx,width=8).pack(side="left");ttk.Spinbox(controls,from_=-500,to=500,increment=.1,textvariable=self.video_overlay_dy,width=8).pack(side="left")
            ttk.Button(controls,text="Auto projected",command=self._overlay_use_auto).pack(side="left",padx=3);ttk.Button(controls,text="Apply to output",command=self._overlay_apply_to_output).pack(side="left",padx=3);ttk.Button(controls,text="Refresh projected view",command=self._prepare_projected_overlay).pack(side="left",padx=3);ttk.Button(controls,text="Reset candidate",command=lambda:(self.video_overlay_dx.set(0),self.video_overlay_dy.set(0),self._refresh_overlay_preview())).pack(side="left",padx=3)
            nudge=ttk.Frame(box);nudge.pack(fill="x",pady=(5,0));ttk.Label(nudge,text="Nudge candidate:").pack(side="left")
            for label,dx,dy in (("←",-1,0),("→",1,0),("↑",0,-1),("↓",0,1),("←0.1",-.1,0),("→0.1",.1,0),("↑0.1",0,-.1),("↓0.1",0,.1)):ttk.Button(nudge,text=label,command=lambda dx=dx,dy=dy:self._overlay_nudge(dx,dy),width=5).pack(side="left",padx=1)
            ttk.Label(nudge,text="LEFT = current fused video-view proxy. RIGHT = aligned candidate fused proxy. Apply to output writes the exact candidate into the final video render.").pack(side="left",padx=(10,0))
            self.video_overlay_canvas=tk.Canvas(box,bg="#111317",highlightthickness=1);self.video_overlay_canvas.pack(fill="both",expand=True,pady=(8,0));self.video_overlay_canvas.bind("<Configure>",lambda e:self._refresh_overlay_preview());self.video_overlay_canvas.bind("<ButtonPress-1>",self._overlay_drag_start);self.video_overlay_canvas.bind("<B1-Motion>",self._overlay_drag_motion);self.video_overlay_window=win
            self.video_overlay_mode.trace_add("write",lambda *_:self._refresh_overlay_preview());self.video_overlay_dx.trace_add("write",lambda *_:self._refresh_overlay_preview());self.video_overlay_dy.trace_add("write",lambda *_:self._refresh_overlay_preview())
        else:self.video_overlay_window.deiconify();self.video_overlay_window.lift()
        try:self._prepare_projected_overlay()
        except Exception as e:messagebox.showerror("Stereo Align",str(e),parent=self.video_overlay_window)

    def _refresh_overlay_preview(self):
        canvas=self.video_overlay_canvas
        if canvas is None or self.video_overlay_window is None or not self.video_overlay_window.winfo_exists():return
        L=self._video_overlay_eye_left;R=self._video_overlay_eye_right
        if L is None or R is None:
            canvas.delete("all")
            canvas.create_text(max(1,canvas.winfo_width())//2,max(1,canvas.winfo_height())//2,
                               text="Preparing projected video alignment preview…",fill="#9aa6b2",
                               font=("TkDefaultFont",12,"bold"))
            return
        cw=max(1,canvas.winfo_width());ch=max(1,canvas.winfo_height());mode=self.video_overlay_mode.get()
        current=make_alignment_overlay_rgb(L,R,self.video_overlay_alpha.get(),mode,0,0)
        candidate=make_alignment_overlay_rgb(L,R,self.video_overlay_alpha.get(),mode,self.video_overlay_dx.get(),self.video_overlay_dy.get())
        h,w=current.shape[:2];gap=6;combo=np.zeros((h,w*2+gap,3),np.uint8);combo[:,:w]=current;combo[:,w+gap:]=candidate
        sc=min((cw-8)/combo.shape[1],(ch-30)/h);self._overlay_view_scale=sc;nw=max(1,round(combo.shape[1]*sc));nh=max(1,round(h*sc));ox=(cw-nw)//2;oy=max(24,(ch-nh)//2)
        im=Image.fromarray(combo).resize((nw,nh),Image.Resampling.LANCZOS);ph=ImageTk.PhotoImage(im);canvas.delete("all");canvas.create_image(ox,oy,anchor="nw",image=ph);self.video_overlay_tk=ph
        canvas.create_text(ox+round(w*sc)/2,8,text="CURRENT FUSED VIDEO VIEW",fill="#d8dde6",anchor="n",font=("TkDefaultFont",11,"bold"));canvas.create_text(ox+round((w+gap+w/2)*sc),8,text="ALIGNED FUSED CANDIDATE",fill="#7ee7ff",anchor="n",font=("TkDefaultFont",11,"bold"));divx=ox+round((w+gap/2)*sc);canvas.create_line(divx,oy,divx,oy+nh,fill="#59616e",width=2)

    def open_output_frame_checker(self):
        """Open one combined playback + frame-accurate inspection window.

        Playback and frame checking share one visual frame/canvas.  Sequential playback reads
        frames in decode order (important for HEVC performance); explicit frame
        changes seek only when the user scrubs, enters an index, or presses the
        previous/next buttons.
        """
        path=self.last_output_path or self.output.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Output inspector","Generate a video first, or choose an existing output path.",parent=self.win);return
        cap=cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Output inspector",f"Could not open generated video:\n{path}",parent=self.win);return
        n=max(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        dlg=tk.Toplevel(self.win);dlg.title("Generated video — unified player / frame inspector");dlg.geometry("1280x820");dlg.minsize(760,520)
        box=ttk.Frame(dlg,padding=8);box.pack(fill="both",expand=True)
        frame_var=tk.IntVar(value=0);info=tk.StringVar(value=f"Frame 0 / {n-1}")
        playing={"value":False};timer={"id":None};holder={"photo":None,"busy":False}
        controls=ttk.Frame(box);controls.pack(fill="x")
        play_btn=ttk.Button(controls,text="▶ Play");play_btn.pack(side="left")
        ttk.Button(controls,text="■ Stop",command=lambda:stop(True)).pack(side="left",padx=4)
        ttk.Button(controls,text="◀ Previous",command=lambda:show(max(0,frame_var.get()-1),True)).pack(side="left",padx=(10,2))
        ttk.Button(controls,text="Next ▶",command=lambda:show(min(n-1,frame_var.get()+1),True)).pack(side="left",padx=2)
        ttk.Label(controls,text="Frame").pack(side="left",padx=(10,3))
        spin=ttk.Spinbox(controls,from_=0,to=n-1,textvariable=frame_var,width=10);spin.pack(side="left")
        ttk.Label(controls,textvariable=info).pack(side="left",padx=8)
        if fps>0:ttk.Label(controls,text=f"{fps:.6g} fps").pack(side="left",padx=8)
        scale=ttk.Scale(box,from_=0,to=max(0,n-1),orient="horizontal");scale.pack(fill="x",pady=(7,5))
        canvas=tk.Canvas(box,bg="#111317",highlightthickness=1);canvas.pack(fill="both",expand=True)

        def draw_frame(frame):
            h,w=frame.shape[:2];cw=max(10,canvas.winfo_width()-8);ch=max(10,canvas.winfo_height()-8)
            sc=min(cw/w,ch/h);dw=max(1,round(w*sc));dh=max(1,round(h*sc));ox=(canvas.winfo_width()-dw)//2;oy=(canvas.winfo_height()-dh)//2
            im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((dw,dh),Image.Resampling.LANCZOS)
            ph=ImageTk.PhotoImage(im);canvas.delete("all");canvas.create_image(ox,oy,anchor="nw",image=ph);holder["photo"]=ph

        def update_labels(idx):
            frame_var.set(idx)
            holder["busy"]=True
            try: scale.set(idx)
            finally: holder["busy"]=False
            info.set(f"Frame {idx} / {n-1}"+(f"   {idx/fps:.3f} s" if fps>0 else ""))

        def stop(reset=False):
            playing["value"]=False;play_btn.configure(text="▶ Play")
            if timer["id"] is not None:
                try:dlg.after_cancel(timer["id"])
                except Exception:pass
                timer["id"]=None
            if reset:
                cap.set(cv2.CAP_PROP_POS_FRAMES,0);ok,frame=cap.read()
                if ok:update_labels(0);draw_frame(frame)

        def show(idx,seek=True):
            stop(False);idx=max(0,min(n-1,int(float(idx))))
            if seek:cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
            ok,frame=cap.read()
            if ok:
                actual=max(0,int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1);update_labels(actual);draw_frame(frame)

        def playback_tick():
            if not playing["value"]:return
            ok,frame=cap.read()
            if not ok:stop(False);return
            idx=max(0,int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1);update_labels(idx);draw_frame(frame)
            timer["id"]=dlg.after(max(1,round(1000.0/(fps if fps>0 else 30.0))),playback_tick)

        def toggle_play():
            if playing["value"]:stop(False);return
            next_idx=min(n-1,frame_var.get()+1);cap.set(cv2.CAP_PROP_POS_FRAMES,next_idx)
            playing["value"]=True;play_btn.configure(text="❚❚ Pause");playback_tick()

        def scale_changed(v):
            if holder["busy"]:return
            show(round(float(v)),True)

        play_btn.configure(command=toggle_play);scale.configure(command=scale_changed)
        spin.bind("<Return>",lambda e:show(frame_var.get(),True))
        # Resize redraws the current frame once; it does not start playback.
        canvas.bind("<Configure>",lambda e:show(frame_var.get(),True))
        def close():
            stop(False);cap.release();dlg.destroy()
        dlg.protocol("WM_DELETE_WINDOW",close);dlg.after_idle(lambda:show(0,True))

    def start_conversion(self):
        if self.worker and self.worker.is_alive():return
        try:
            pts=self._read_coords();li=video_probe(self.left_video.get());ri=video_probe(self.right_video.get())
            lp=self._profile("L",li["width"],li["height"]);rp=self._profile("R",ri["width"],ri["height"])
            args=dict(left_path=self.left_video.get(),right_path=self.right_video.get(),output_path=self.output.get(),lp=lp,rp=rp,
                      left_a=pts["A"][0],left_b=pts["B"][0],right_a=pts["A"][1],right_b=pts["B"][1],
                      left_start=self.left_start.get(),left_end=self.left_end.get(),right_start=self.right_start.get(),right_end=self.right_end.get(),
                      output_width=self.output_width.get(),roll=self.roll.get(),pitch=self.pitch.get(),yaw=self.yaw.get(),
                      codec=self.codec.get(),fps_mode=self.fps_mode.get(),sampling=self.sampling.get(),length_policy=self.length_policy.get(),
                      right_shift_x_deg=self.stereo_x.get(),right_shift_y_deg=self.stereo_y.get())
        except Exception as e:messagebox.showerror("Video conversion",str(e),parent=self.win);return
        self.cancel_event.clear();self.progress.set(0);self.start_btn.configure(state="disabled");self.cancel_btn.configure(state="normal")
        def run():
            try:
                def cb(done,total,stage,frac):self.win.after(0,lambda f=frac,s=stage:(self.progress.set(100*f),self.status.set(s)))
                result=convert_stereo_video(**args,progress=cb,stop_requested=self.cancel_event.is_set)
                self.win.after(0,lambda r=result:self._finished(r))
            except InterruptedError:self.win.after(0,lambda:self._failed("Conversion cancelled."))
            except Exception as e:self.win.after(0,lambda e=e:self._failed(str(e)))
        self.worker=threading.Thread(target=run,daemon=True);self.worker.start()
    def cancel_conversion(self):self.cancel_event.set();self.status.set("Cancelling after the current frame…")
    def _finished(self,r):
        self.progress.set(100);self.status.set(f"Finished {r['frames']} frames -> {r['output']} ({r['width']}×{r['height']}, {r['fps']:.3f} fps, {r['codec']}, timing={r.get('fps_mode','source')}, length={r.get('length_policy','strict')})")
        self.last_output_path=r.get("output",self.output.get())
        self.start_btn.configure(state="normal");self.cancel_btn.configure(state="disabled")
        if self.output_check_btn is not None:self.output_check_btn.configure(state="normal")
        messagebox.showinfo("Video conversion","Finished video conversion. Use Inspect generated video for playback and frame-by-frame checking.",parent=self.win)
    def _failed(self,msg):
        self.start_btn.configure(state="normal");self.cancel_btn.configure(state="disabled");self.status.set(msg)
        if msg!="Conversion cancelled.":messagebox.showerror("Video conversion",msg,parent=self.win)


def launch_gui(left=None,right=None):
    if tk is None:
        raise SystemExit("Tkinter/Pillow are required for the GUI. Install python3-tk and pillow.")
    root=tk.Tk(); app=AugenGUI(root,left,right); root.mainloop()

def parser():
    q=argparse.ArgumentParser(description='180pyugen v35 — still-image + video VR180'); q.add_argument('--ini',default=None, help='Optional external 180Augen/180Kino.ini; embedded profiles are used by default'); s=q.add_subparsers(dest='cmd',required=True)
    x=s.add_parser('profiles'); x.set_defaults(fn=cmd_profiles)
    x=s.add_parser('info'); x.add_argument('--profile',required=True); x.set_defaults(fn=cmd_info)
    x=s.add_parser('test'); x.add_argument('--profile',required=True); x.add_argument('--width',type=int,default=4000); x.add_argument('--height',type=int,default=3000); x.set_defaults(fn=cmd_test)
    x=s.add_parser('synthetic'); x.add_argument('--profile',required=True); x.add_argument('--width',type=int,default=1200); x.add_argument('--height',type=int,default=1200); x.add_argument('--output',default='synthetic.png'); x.set_defaults(fn=cmd_synth)
    x=s.add_parser('match'); x.add_argument('--left',required=True); x.add_argument('--right',required=True); x.add_argument('--x',type=float,required=True); x.add_argument('--y',type=float,required=True); x.add_argument('--template',type=int,default=20); x.set_defaults(fn=cmd_match)
    x=s.add_parser('render'); x.add_argument('--left',required=True); x.add_argument('--right',required=True); x.add_argument('--left-profile',required=True); x.add_argument('--right-profile',required=True); 
    for n in ('left-ax','left-ay','left-bx','left-by','right-ax','right-ay','right-bx','right-by'): x.add_argument('--'+n,type=float,required=True)
    x.add_argument('--height',type=int,default=512); x.add_argument('--roll',type=float,default=0); x.add_argument('--pitch',type=float,default=0); x.add_argument('--yaw',type=float,default=0); x.add_argument('--output',default='vr180_test.png'); x.set_defaults(fn=cmd_render)
    x=s.add_parser('render-exe', help='Render using the reconstructed 180Augen L/R square-hemisphere geometry'); x.add_argument('--left',required=True); x.add_argument('--right',required=True); x.add_argument('--left-profile',required=True); x.add_argument('--right-profile',required=True)
    for n in ('left-ax','left-ay','left-bx','left-by','right-ax','right-ay','right-bx','right-by'): x.add_argument('--'+n,type=float,required=True)
    x.add_argument('--height',type=int,default=1024); x.add_argument('--roll',type=float,default=0); x.add_argument('--pitch',type=float,default=0); x.add_argument('--yaw',type=float,default=0); x.add_argument('--output',default='LROut_reconstructed.jpg'); x.set_defaults(fn=cmd_render_exe)
    x=s.add_parser('projection'); x.add_argument('--profile',required=True); x.add_argument('--samples',type=int,default=101); x.set_defaults(fn=cmd_projection)
    x=s.add_parser('stereo-exe'); x.add_argument('--left-profile',required=True); x.add_argument('--right-profile',required=True)
    for n in ('lw','lh','rw','rh','lax','lay','lbx','lby','rax','ray','rbx','rby'): x.add_argument('--'+n,type=float,required=True)
    x.set_defaults(fn=cmd_stereo_exe)
    x=s.add_parser('gopro'); x.add_argument('--width',type=int,default=3840); x.add_argument('--height',type=int,default=2160); x.add_argument('--fov',type=float,default=177.0); x.set_defaults(fn=cmd_gopro)
    x=s.add_parser('video', help='180Kino-style stereo video conversion')
    x.add_argument('--left',required=True); x.add_argument('--right',required=True); x.add_argument('--output',default='VROut.mp4')
    x.add_argument('--left-profile',required=True); x.add_argument('--right-profile',required=True)
    for n in ('left-ax','left-ay','left-bx','left-by','right-ax','right-ay','right-bx','right-by'): x.add_argument('--'+n,type=float,required=True)
    x.add_argument('--left-start',type=int,default=0); x.add_argument('--left-end',type=int,required=True)
    x.add_argument('--right-start',type=int,default=0); x.add_argument('--right-end',type=int,required=True)
    x.add_argument('--width',type=int,default=4096); x.add_argument('--roll',type=float,default=0); x.add_argument('--pitch',type=float,default=0); x.add_argument('--yaw',type=float,default=0)
    x.add_argument('--codec',default='auto'); x.add_argument('--fps',type=float,default=None); x.add_argument('--fps-mode',choices=('kino','source'),default='kino'); x.add_argument('--sampling',choices=('fast','hq','native'),default='fast'); x.add_argument('--length-policy',choices=('strict','trim','repeat_last','black'),default='strict'); x.set_defaults(fn=cmd_video)
    x=s.add_parser('clip-video', help='Extract synchronized JPEG reference frames')
    x.add_argument('--left',required=True); x.add_argument('--right',required=True); x.add_argument('--output-dir',default='cut_img')
    x.add_argument('--left-start',type=int,required=True); x.add_argument('--left-end',type=int,required=True); x.add_argument('--right-start',type=int,required=True); x.add_argument('--right-end',type=int,required=True)
    x.add_argument('--jpeg-quality',type=int,default=95); x.set_defaults(fn=cmd_clip_video)
    x=s.add_parser('gui', help='Launch the graphical 180pyugen UI'); x.add_argument('--left',default=None); x.add_argument('--right',default=None); x.set_defaults(fn=lambda a: launch_gui(a.left,a.right)); return q

def main():
    a=parser().parse_args()
    if a.cmd in ('match','render','synthetic','stereo-exe','video','clip-video') and cv2 is None: raise SystemExit('This command requires opencv-python')
    a.fn(a)
if __name__=='__main__': main()
