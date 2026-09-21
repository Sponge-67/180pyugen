# 180pyugen Web v2.2 - image + video Python server edition

This is the server-backed web version of **180pyugen**. It now contains two independent workflows inside the same application:

- **Images** - reconstructed 180Augen-style still-image VR180 conversion.
- **Video** - 180Kino-style synchronized stereo-video VR180 conversion.

The browser handles ordinary-user interaction (file selection, preview, A/B picking, synchronization settings and progress). The high-accuracy projection/stereo math runs in Python/OpenCV on server workers.

## Ordinary-user image workflow

1. Open the web page and choose **Images**.
2. Load left/right fisheye images.
3. Pick A/B on the large left view. The server uses the reconstructed 20x20 `TM_SQDIFF` matcher to find each right point.
4. Press **Render VR180 image**.
5. Preview/download JPEG or PNG.

## Ordinary-user video workflow

1. Choose **Video**.
2. Select left/right movies and click **Upload / inspect pair**.
3. Enter synchronized left/right reference frame numbers and load those frames.
4. Pick A/B on the large left reference frame; the server finds the corresponding point in the right reference frame.
5. Set equal-length left/right frame ranges, output width, profiles and optional zenith correction.
6. Press **Start video conversion**.
7. Preview/download the resulting VR180 movie.

The video page also includes a **JPEG clipping ZIP** tool for extracting `L<frame>.jpg` / `R<frame>.jpg` reference images from selected frame ranges.

The default video renderer precomputes the fixed fisheye-to-equirectangular source maps once and reuses them for every synchronized frame pair. `Fast OpenCV` uses `cv2.remap`; `Native-style exact` uses the validated 180Augen-like bilinear sampler and is intended for parity testing rather than speed.

Audio is currently not copied into the output video, matching the documented 180Kino 2.0 behavior. It can be muxed later with FFmpeg without re-rendering the video frames.

## Start the server

### Easiest local development test

```bash
./start-dev.sh
```

On first launch the script creates a local `.venv`, installs `requirements.txt`, enables the inline single-worker development mode, and starts:

```text
http://127.0.0.1:8080
```

If virtual-environment creation fails on Debian/Ubuntu:

```bash
sudo apt install python3-venv
```

### Docker / normal server deployment

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8080
```

For more simultaneous users:

```bash
docker compose up --build --scale worker=2
```

## Architecture

```text
Browser UI
   |
   +-- image session / match / render
   |
   +-- video session / reference frames / match / clip / render
   |
   v
FastAPI
   |
   +-- inline worker in development
   |
   +-- Redis/RQ warm worker(s) in normal deployment
            |
            v
        180pyugen Python engine
        NumPy + OpenCV
            |
            +-- JPEG/PNG result
            +-- MP4/AVI/MOV result
            +-- clipped JPEG ZIP
```

## Video implementation notes

The integrated video core is based on the reverse-engineered still-image geometry plus the supplied 180Kino 2.0 materials. The original binary/PDB shows an OpenCV/CUDA implementation with GPU video readers/writer and a custom `Fish2Eqtngr` CUDA kernel. This v2.2 server build is intentionally portable: it uses ordinary OpenCV video I/O and CPU rendering, while retaining the same camera/lens profile model and 3-D A/B alignment architecture.

The `180Kino.ini` parser is also supported. Native camera records contain **17 floating slots** before the six integer fields: radius magnification, projection k, three interval boundaries and four `(A,B,C)` coefficient triples. This is important for the 180Kino EM10 entry, where magnification `1.58` and projection `k=0.83` are separate values.

Exact instruction-level parity with the original 180Kino CUDA kernel is **not claimed yet**. A real source-video/output pair from 180Kino can be used for the next parity pass.

## Temporary data / privacy

Inputs and results live under the shared Docker volume `/data/180pyugen`. Old sessions/results are deleted automatically (2 hours by default). Still-image input deletion after render is enabled by default in the UI; video deletion is optional because repeated synchronization/calibration tests often reuse the uploaded movies.

Environment knobs:

- `PYUGEN_MAX_UPLOAD_MB` - still-image per-file upload cap (default 120 MB)
- `PYUGEN_MAX_VIDEO_UPLOAD_MB` - video per-file upload cap (default 4096 MB)
- `PYUGEN_MAX_OUTPUT_HEIGHT` - maximum per-eye height; 4096 permits 8192x4096 SBS video
- `PYUGEN_SESSION_TTL` - session lifetime in seconds
- `PYUGEN_RESULT_TTL` - result lifetime in seconds
- `REDIS_URL` - queue Redis URL

## Current profiles

The embedded engine includes the validated 180Augen profiles, the calibrated GoPro HERO12 + Max Lens Mod 2.0 L/R profiles, and the camera/lens records from the supplied 180Kino configuration (DJI Action2, 180Kino EM10, DJI Mini2/3 and DJI Action5 variants).

Desktop **180pyugen v22** and this web v2.2 use the same Python engine so image/video math changes do not need to be maintained independently.


## 180Kino tutorial compatibility update (v2.3)

The supplied original `L.mp4` / `R.mp4` sample pair is 4096x3072 HEVC at
60000/1001 fps (600 frames each). The tutorial preset uses reference frames
L180 / R186 and conversion ranges L180..480 / R186..486 with the `DJI Action2`
profile.

Two native 180Kino-specific differences are now preserved separately from the
180Augen still-image workflow:

* mode-3 camera profiles use the documented piecewise quadratic
  `y = a*theta^2 + b*theta + c`;
* Get A/B uses a 20x20 `TM_SQDIFF_NORMED` template match (OpenCV method 1),
  while the reconstructed 180Augen still workflow keeps `TM_SQDIFF` (method 0).


## v2.4 overlay preview

A visual alignment mode was added for both still images and video reference frames. It draws the right view first and blends the left view on top with adjustable opacity (default 50%). This makes it easier to inspect whether the two views line up on the same scene features before rendering.


## v2.5 native 180Kino output compatibility

The supplied original VROut establishes that the tutorial conversion is 8192x4096 HEVC Main, yuv420p, exactly 60 fps, and 301 frames for the inclusive L180..480 / R186..486 ranges. The supplied cut_img archive also confirms the tutorial clipping ranges and OpenCV-style quality-95 JPEG output.

Video conversion now has a native-like HEVC option through FFmpeg (`libx265`) plus explicit `hevc-nvenc`, and a timing selector. `180Kino rounded FPS` rounds the source FPS before encoding (59.9400599 -> 60.0 in the tutorial sample); `Preserve source FPS` retains the source rate. The tutorial preset selects HEVC and native timing. The web clipping ZIP stores files under `cut_img/`.


## v2.7 Stereo Align + quality/performance

The alignment inspector now includes BorisFX-style negative/invert-and-mix mode, alpha blend, absolute-style difference, and red/cyan inspection, plus manual right-view X/Y offsets and A/B-derived automatic preview offsets. Final rendering has independent right-eye convergence and vertical trim in VR degrees. Video sampling adds a Lanczos4 high-quality mode; fast bilinear mode converts the maps once to OpenCV fixed-point remap maps to reduce memory and improve throughput; eye-level parallelism is enabled only when OpenCV itself is single-threaded. Native-style exact sampling remains available for parity testing.

Overlay previews can also be click-dragged directly to move the right view, matching the manual alignment interaction style used by professional stereo tools.


## v2.7 reference-display fix and documentation

### Automatic video reference display

Selecting the second movie now automatically creates/probes the temporary video
session and loads the current left/right reference frames into the browser.
The explicit **Load reference frames** button remains available when the user
changes either frame number.  This removes the previous ambiguous state where
valid movie paths were loaded but the reference canvases remained blank.

### Processing architecture

The project is intentionally split into a thin interface layer and one shared
Python geometry engine:

```text
Still image
  decode L/R
    -> camera/lens profile
    -> A/B pixels to 3-D rays
    -> two-stage right-to-left rotation
    -> 180x180-degree output ray field
    -> zenith rotation
    -> ray-to-fisheye source coordinates
    -> sampling
    -> SBS pack
    -> PNG/JPEG

Video
  upload/open L/R movies
    -> probe streams
    -> synchronized reference frames
    -> A/B matching
    -> build stereo transform once
    -> build left/right source maps once
    -> sequential synchronized frame loop
         decode -> remap -> optional right-eye trim -> pack -> encode
    -> MP4/HEVC result
```

The browser does not maintain a second independent projection implementation.
It handles interaction and previews; FastAPI/Python performs matching and final
rendering.  This is important for parity work because any geometry correction
made in the desktop core immediately applies to the web workers as well.

### Quality/performance modes

* **Fast**: OpenCV remap, optimized maps; normal production choice.
* **HQ**: Lanczos4 interpolation; slower, useful for fine detail.
* **Native**: reconstructed native-style interpolation; intended mainly for
  comparison and reverse-engineering validation.

HEVC/H.265 is the native-oriented output choice.  NVENC can offload encoding on
supported NVIDIA hardware, but GPU video encoding and fisheye remapping are
separate stages.

### Stereo alignment model

The source overlay is diagnostic.  The true stereo correction comes from A/B
pixels being converted into 3-D camera rays and solving the relative camera
rotation.  Small final global disparities can be corrected with right-eye
convergence/vertical trim in VR degrees.  Large trims normally signal poor A/B
selection or an inaccurate camera/lens profile.

### In-app information page

The web header now includes **Info / mechanics**, and the desktop application
has an **Info / mechanics…** window.  These pages document the still pipeline,
video pipeline, lens model, A/B matching, Stereo Align tools, quality modes,
server architecture, and current native-parity limitations.


## v2.8 video review and UI options

- Reference frames are visible in the main video workspace and can be stepped as a synchronized pair with Previous/Next controls.
- A generated-video frame checker can seek the finished result by zero-based frame index and draw that exact decoded frame to a canvas for inspection.
- The web UI now supports Dark/Light themes and Focus/Equal-split workspace layouts. Preferences persist in browser local storage.
- The older explanatory sentence about video being a separate work process was removed from the operational page; pipeline documentation remains in Info / mechanics.


## v2.9 reference-view and playback changes

- Focus layout was removed; preview panes are always equal width.
- **Info / mechanics** was renamed to **Info**.
- Video reference-frame canvases now follow the decoded frame aspect ratio, removing the oversized top/bottom black bands.
- The generated-video player and per-frame checker now share one inspector panel.
- The full 4K/8K render is not loaded into the browser player. A 1280-wide H.264 `+faststart` proxy is encoded concurrently from already-rendered frames for smooth playback and seeking, while Download still returns the untouched full-resolution result.


## v2.10 projected Stereo Align

Stereo Align now compares the final projected equirectangular eyes after camera/lens mapping, A/B 3-D rotation, zenith correction and current convergence/vertical trim. The left side of the inspector shows the current output alignment; the right side shows a candidate right-eye translation. Applying the candidate converts its preview-pixel shift directly into the final right-eye degree trim, so an alignment accepted in the inspector is the same correction used by the real render.

The generated-video player and per-frame checker now share one viewport. Playback displays the H.264 proxy directly; selecting a frame pauses playback and overlays that exact decoded frame in the same visual surface.


## v2.11 alignment-preview display fix

The projected Stereo Align inspector no longer presents an unexplained black canvas while no projected bitmap has been loaded. Reference-frame loading automatically requests the final-eye preview, a visible loading/error message is rendered into the canvas, PNG decoding has an HTMLImage fallback for browsers where `createImageBitmap()` fails, stale asynchronous preview responses are ignored, and generated preview responses are marked `no-store`. The right-hand **ALIGNED CANDIDATE** now starts at the phase-correlation residual shift while the left-hand **CURRENT** panel remains the actual unmodified current output.


## v2.12 fused alignment preview

The Stereo Align window now defaults to **Perceptual fuse**. This is a best-effort flat-screen proxy of the final projected stereo pair after the current candidate correction is applied. Adjusting the candidate and pressing **Apply to output** writes those values directly into the final convergence/vertical-trim controls used by rendering.
