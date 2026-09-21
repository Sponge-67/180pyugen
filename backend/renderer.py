"""Memory-bounded web still renderer.

The web service renders large images in horizontal row blocks.  Blocking does
not change the projection mathematics; it merely avoids allocating the full
N x N x 3 floating-point direction field for an 8K-class job at once and gives
the job system natural progress checkpoints.
"""
from __future__ import annotations
import math
import numpy as np
from . import engine


def render_stereo_blocked(left, right, lp, rp, left_a, left_b, right_a, right_b,
                          output_height=1024, roll=0.0, pitch=0.0, yaw=0.0,
                          block_rows=32, progress=None):
    """Memory-bounded equivalent of engine.render_stereo_exe_geometry().

    Rendering in row blocks keeps memory practical for 2K/4K web jobs and also
    provides progress callbacks for the web UI. The mathematical path is the
    same reconstructed 180Augen pipeline used by the desktop branch.
    """
    n = int(output_height)
    la = engine.pixel_to_ray(*left_a, lp, left.shape[1], left.shape[0])
    lb = engine.pixel_to_ray(*left_b, lp, left.shape[1], left.shape[0])
    ra = engine.pixel_to_ray(*right_a, rp, right.shape[1], right.shape[0])
    rb = engine.pixel_to_ray(*right_b, rp, right.shape[1], right.shape[0])
    R = engine.stereo_rotation(la, lb, ra, rb)
    Z = engine.zenith_rotation(roll, pitch, yaw)

    out = np.zeros((n, 2*n, 3), dtype=np.uint8)
    x = np.arange(n, dtype=np.float64)
    lon = x * (math.pi / n) - math.pi/2.0
    sx = np.sin(lon)
    cx = np.cos(lon)

    for y0 in range(0, n, int(block_rows)):
        y1 = min(n, y0 + int(block_rows))
        y = np.arange(y0, y1, dtype=np.float64)
        lat = math.pi/2.0 - y * (math.pi/n)
        sl = np.sin(lat)[:, None]
        cl = np.cos(lat)[:, None]

        bh = y1 - y0
        d = np.empty((bh, n, 3), dtype=np.float64)
        d[:,:,0] = cl * sx[None,:]
        d[:,:,1] = sl
        d[:,:,2] = cl * cx[None,:]
        flat = d.reshape(-1, 3)

        dl = flat @ Z.T
        dr = dl @ R
        L = engine.exe_sample_camera_directions(left, lp, dl).reshape(bh, n, 3)
        RR = engine.exe_sample_camera_directions(right, rp, dr).reshape(bh, n, 3)
        out[y0:y1, :n] = L
        out[y0:y1, n:] = RR

        if progress:
            progress(y1 / n)
    return out
