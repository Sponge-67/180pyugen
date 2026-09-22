from __future__ import annotations
from dataclasses import asdict
from . import engine

GOPRO_L = engine.GOPRO_CAL_L_NAME
GOPRO_R = engine.GOPRO_CAL_R_NAME
GOPRO_APPROX = "GoPro HERO12 Black + Max Lens Mod 2.0 (177deg approx)"


def profile_names() -> list[str]:
    names = [p.name for p in engine.parse_profiles(None)]
    for n in (GOPRO_L, GOPRO_R, GOPRO_APPROX):
        if n not in names:
            names.append(n)
    return names


def resolve_profile(name: str, side: str, width: int, height: int):
    if name == GOPRO_L:
        return engine.gopro12_max_lens_mod2_calibrated_profile("L", width, height)
    if name == GOPRO_R:
        return engine.gopro12_max_lens_mod2_calibrated_profile("R", width, height)
    if name == GOPRO_APPROX:
        return engine.gopro12_max_lens_mod2_profile(width, height, 177.0)
    return engine.profile_by_name(engine.parse_profiles(None), name)


def profile_info(name: str, side: str, width: int, height: int) -> dict:
    p = resolve_profile(name, side, width, height)
    cx, cy = engine.calculate_optical_axis(p, width, height)
    radius = engine.calculate_radius(p, width, height)
    info = {
        "name": p.name,
        "radius": float(radius),
        "axis": [float(cx), float(cy)],
        "projection_mode": int(p.projection.mode),
        "k": float(p.projection.k),
    }
    if name in (GOPRO_L, GOPRO_R):
        ok, warning = engine.gopro_calibration_compatibility(width, height)
        info["calibration_compatible"] = bool(ok)
        info["calibration_warning"] = "" if ok else warning
    return info
