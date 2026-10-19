"""
Vanish effect v6 (Windows) - Thanos "snap" & smooth "fade" for virtual webcams.

GLOBAL HOTKEYS (work while Zoom / Meet / Teams / OBS has focus):
  Ctrl+Alt+B   Capture clean background (3 s countdown -> STEP OUT of frame)
  Ctrl+Alt+V   Snap away / reappear
  Ctrl+Alt+E   Switch effect: SNAP (dust) <-> FADE (dissolve)
  Ctrl+Alt+C   Webcam driver settings dialog
  Ctrl+Alt+S   Pause / resume finger-snap listening (when started with --snap)

PREVIEW WINDOW KEYS:  b = background | space = snap | e = effect | m = show mask | q = quit

INSTALL:   pip install opencv-python mediapipe==0.10.14 pyvirtualcam numpy
CHECK PC:  python vanish.py --selftest        (no camera needed; prints your ms/frame)
OPTIONS:   --cam 1  --res 960x540  --fps 30  --no-preview  --mode fade
SNAP:      pip install sounddevice  ->  python vanish.py --snap   (tune: --snap-test, --snap-sens 1-10, --mic N)

What changed vs v5.2 (why it was laggy / didn't fully vanish):
  * Compositing used full-res float32 NumPy math (~40-90 ms/frame). Now cv2.blendLinear on only
    the bounding box of the effect, with float32 maths at 320x180.
  * The person mask came only from an async model that is 2-4 frames stale, heavily dilated and
    thresholded. Now: model (interior) + clean-plate difference (exact, current-frame edges).
  * Webcam auto-exposure/white-balance made the replaced region visibly different from the
    live image. Now the clean plate is gain-matched to the live frame continuously.
  * Main loop slept after each frame and could skip camera frames (judder). Now event driven.
  * Segmentation errors were silently swallowed. Now they are printed.
"""

import os

# Must be set BEFORE importing cv2: makes the MSMF webcam backend open in ~1s instead of ~20s.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import argparse
import ctypes
import math
import queue
import sys
import threading
import time

import cv2
import numpy as np

# ------------------------------------------------------------------ Settings
CAP_W, CAP_H, TARGET_FPS = 1280, 720, 30
PROC_W, PROC_H = 1280, 720          # output size; must be multiples of 20
CAMERA_INDEX = 0
PREVIEW = True
PREVIEW_EVERY = 3                   # draw preview every N frames

ANALYSIS_W = 320                    # mask / particle grid width (height follows aspect)
SEG_THRESH_CORE = 0.60              # model prob treated as "certainly person"
SEG_THRESH_ROI = 0.20               # model prob that opens the region where edges may refine
SEG_EVERY = 2                       # submit a frame to the model every N frames
ROI_GROW = 25                       # grid-px kernel; lets bg-difference edges move this far past a stale model mask
DIFF_LO_MIN, DIFF_LO_MAX, DIFF_RAMP = 10.0, 24.0, 18.0   # adaptive: ramp starts at ~3.5x measured noise
MIN_MATCH_FRACTION = 0.15           # scene counts as 'background still valid' if >=15% of frame matches it
GAIN_CELL = 8                       # lighting map is fitted on 8x8 grid-px cells
MASK_SMOOTH_ALPHA = 0.6             # temporal smoothing (higher = more responsive)
BG_SETTLE_FRAMES = 12               # discard frames right after countdown (auto-exposure settles)
BG_FRAMES = 30                      # frames averaged into the clean plate

EFFECT = "snap"
FADE_SECONDS = 0.6
SNAP_SECONDS = 2.2
SPREAD = 0.40
DRIFT_PX = 65                       # at 640 px width; scaled automatically

IS_WIN = sys.platform == "win32"
BACKENDS = [("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY), ("DSHOW", cv2.CAP_DSHOW)] if IS_WIN \
    else [("ANY", cv2.CAP_ANY)]


def smoothstep(x):
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def apply_windows_optimizations():
    if not IS_WIN:
        return
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00008000)  # ABOVE_NORMAL
    except Exception:
        pass


def end_windows_optimizations():
    if IS_WIN:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass


def fit_frame(frame, w, h):
    """Center-crop to the target aspect (no stretching), then resize."""
    fh, fw = frame.shape[:2]
    if fw * h != fh * w:
        if fw * h > fh * w:                       # too wide
            nw = fh * w // h
            x0 = (fw - nw) // 2
            frame = frame[:, x0:x0 + nw]
        else:                                     # too tall
            nh = fw * h // w
            y0 = (fh - nh) // 2
            frame = frame[y0:y0 + nh]
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return frame


# ------------------------------------------------------------------ Camera
def open_camera(preferred_idx, w, h, fps, backend=None):
    backends = [b for b in BACKENDS if backend in (None, "auto", b[0].lower())] or BACKENDS
    indices = [preferred_idx] + [i for i in range(4) if i != preferred_idx]
    for idx in indices:
        for name, api in backends:
            for mjpg in (True, False):
                cap = cv2.VideoCapture(idx, api)
                if not cap.isOpened():
                    cap.release()
                    continue
                if mjpg:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, fps)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # smaller driver queue = lower latency
                except Exception:
                    pass
                good = False
                for _ in range(20):
                    ok, f = cap.read()
                    if ok and f is not None and f.size > 0:
                        good = True
                        break
                    time.sleep(0.03)
                if good:
                    print(f"[+] Camera: index={idx} backend={name} MJPG={mjpg} "
                          f"{int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.0f}fps")
                    return cap
                cap.release()
    return None


class ThreadedGrabber(threading.Thread):
    """Reads the camera as fast as it delivers; consumer always gets the newest frame."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.running = True

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            if ok and f is not None and f.size > 0:
                with self.cond:
                    self.frame = f
                    self.seq += 1
                    self.cond.notify_all()
            else:
                time.sleep(0.005)

    def wait_new(self, last_seq, timeout=0.25):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout)
            return self.frame, self.seq

    def stop(self):
        self.running = False
        time.sleep(0.05)
        if self.cap.isOpened():
            self.cap.release()


# ------------------------------------------------------------------ Segmentation (async)
class SegWorker(threading.Thread):
    """Runs the MediaPipe model off the render thread. Latest-wins, never blocks rendering."""

    def __init__(self, segmenter):
        super().__init__(daemon=True)
        self.seg = segmenter
        self.ev = threading.Event()
        self.lock = threading.Lock()
        self.inp = None
        self.out = None
        self.busy = False
        self.running = True
        self.last_error = None
        self.start()

    def submit(self, rgb):
        with self.lock:
            if not self.busy:
                self.inp = rgb
                self.busy = True
                self.ev.set()

    def run(self):
        while self.running:
            self.ev.wait()
            self.ev.clear()
            if not self.running:
                break
            with self.lock:
                rgb, self.inp = self.inp, None
            if rgb is None:
                with self.lock:
                    self.busy = False
                continue
            try:
                prob = np.asarray(self.seg.process(rgb).segmentation_mask, dtype=np.float32)
                with self.lock:
                    self.out = prob
            except Exception as e:  # never swallow silently
                msg = repr(e)
                if msg != self.last_error:
                    print(f"[!] Segmentation error: {msg}")
                    self.last_error = msg
            finally:
                with self.lock:
                    self.busy = False

    def get_mask(self):
        with self.lock:
            return self.out

    def clear(self):
        with self.lock:
            self.out = None

    def stop(self):
        self.running = False
        self.ev.set()


# ------------------------------------------------------------------ Effect engine
class VanishEngine:

    def __init__(self, width=PROC_W, height=PROC_H, use_model=True):
        assert width % 4 == 0 and height % 4 == 0, "frame size must be divisible by 4"
        self.PW, self.PH = width, height
        self.AW = ANALYSIS_W if width >= ANALYSIS_W and width % ANALYSIS_W == 0 else width // 4
        self.AH = height * self.AW // width
        self.sx, self.sy = width / self.AW, height / self.AH

        self.worker = None
        self.model_ok = False
        self.model_error = ""
        if use_model:
            try:
                import mediapipe as mp
                seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
                self.worker = SegWorker(seg)
                self.model_ok = True
            except Exception as e:
                self.model_error = repr(e)
                print(f"[!] MediaPipe unavailable ({e!r}). Falling back to background-subtraction only: "
                      f"works best with a static camera and good lighting.")

        self.mode = EFFECT
        self.bg = self.bg_adj = self.bg_small = self.bg_small_adj = None
        self.Gs = np.ones((self.AH, self.AW, 3), np.float32)       # local lighting gain map (live / clean plate)
        self.G_applied = self.Gs.copy()
        self.bgf = None
        self._adj_busy = False
        self.status = ""
        self.alpha = 0.0
        self.last_update_time = time.monotonic()
        self.smooth = None
        self.tick = 0
        self.idle_tick = 0
        self.far_ok = None
        self.last_cover = None
        self.coverage = 0.0
        self._warned_scene = False

        # polynomial basis for the lighting map (cubic surface over the frame)
        self.ch, self.cw = self.AH // GAIN_CELL, self.AW // GAIN_CELL
        def basis(h, w):
            u = (np.arange(w, dtype=np.float64) + 0.5) / w * 2 - 1
            v = (np.arange(h, dtype=np.float64) + 0.5) / h * 2 - 1
            U, V = np.meshgrid(u, v)
            U, V = U.ravel(), V.ravel()
            return np.stack([np.ones_like(U), U, V, U * U, U * V, V * V, U ** 3, U * U * V, U * V * V, V ** 3], 1)
        self.A_cells = basis(self.ch, self.cw)
        self.A_grid = basis(self.AH, self.AW).astype(np.float32)
        self.bg_cells = None

        self.k_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.k_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ROI_GROW, ROI_GROW))
        self.k_edge = np.ones((3, 3), np.uint8)

        AW, AH = self.AW, self.AH
        rng = np.random.default_rng(42)
        s = AW / 640.0
        self.n_a = rng.random((AH, AW), dtype=np.float32)
        self.n_b = rng.random((AH, AW), dtype=np.float32)
        self.n_mid = self._norm(cv2.GaussianBlur(rng.random((AH, AW), dtype=np.float32), (0, 0), 4 * s))
        self.n_low = self._norm(cv2.GaussianBlur(rng.random((AH, AW), dtype=np.float32), (0, 0), 20 * s))
        self.X = np.tile(np.arange(AW, dtype=np.float32), (AH, 1))
        self.Y = np.tile(np.arange(AH, dtype=np.float32)[:, None], (1, AW))
        self.front = (0.75 * (self.X / np.float32(AW)) + 0.25 * self.n_low).astype(np.float32)
        self.noise_mid = (0.5 + self.n_mid).astype(np.float32)
        self.g4 = np.empty((AH, AW, 4), np.uint8)
        self.drift = np.float32(DRIFT_PX * (AW / 640.0))
        self.inv_spread = np.float32(1.0 / SPREAD)

    @staticmethod
    def _norm(a):
        return ((a - a.min()) / (a.max() - a.min() + 1e-6)).astype(np.float32)

    # ---- background
    def set_background(self, bg_frame):
        bg = fit_frame(bg_frame, self.PW, self.PH)
        self.bg = np.ascontiguousarray(bg)
        self.bg_adj = self.bg.copy()
        self.bg_small = cv2.resize(self.bg, (self.AW, self.AH), interpolation=cv2.INTER_AREA)
        self.bg_small_adj = self.bg_small.copy()
        self.bgf = self.bg.astype(np.float32)
        self.bg_small_f = self.bg_small.astype(np.float32)
        self.bg_cells = cv2.resize(self.bg_small, (self.cw, self.ch), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1, 3)
        self.Gs[:] = 1.0
        self.G_applied = self.Gs.copy()
        self.smooth = None
        self.far_ok = None
        self.alpha = 0.0
        self._warned_scene = False
        self.status = ""

    @property
    def bg_ready(self):
        return self.bg is not None

    def _fit_gain(self, small):
        """Robustly fit a smooth per-channel lighting map (live / clean plate). Objects are outliers."""
        ch, cw = self.ch, self.cw
        sm = cv2.resize(small, (cw, ch), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1, 3)
        bm = self.bg_cells
        r = (sm + 2.0) / (bm + 2.0)
        far = None
        if self.far_ok is not None:
            far = cv2.resize(self.far_ok.astype(np.uint8) * 255, (cw, ch), interpolation=cv2.INTER_AREA).reshape(-1) > 230
            if int(far.sum()) < 60:
                far = None
        A = self.A_cells
        reg = np.diag([0.0] + [1.0] * 9)
        Gc = np.empty((ch, cw, 3), np.float32)
        for c in range(3):
            valid = (bm[:, c] > 25) & (bm[:, c] < 235)
            if far is not None:
                valid &= far
            if int(valid.sum()) < 60:
                return False
            rc = r[:, c].astype(np.float64)
            coef = np.zeros(10)
            coef[0] = np.median(rc[valid])
            w = valid.astype(np.float64)
            for floor, cap in ((0.10, 0.35), (0.07, 0.30), (0.05, 0.25), (0.04, 0.20), (0.03, 0.18)):   # tolerance anneals
                e = rc - A @ coef
                sc = max(1.4826 * np.median(np.abs(e[valid])), floor)
                u = e / (4.0 * sc)
                w = valid * np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0) * (np.abs(e) < cap)
                sw = float(w.sum())
                if sw < 25:
                    break
                coef = np.linalg.solve(A.T @ (A * w[:, None]) + 0.01 * sw * reg, A.T @ (w * rc))
            # final map: trusted cells smoothed locally; hidden (occluded) areas diffuse smoothly from visible neighbours
            Wm, Rm = w.reshape(ch, cw).astype(np.float32), rc.reshape(ch, cw).astype(np.float32)
            WR = Rm * Wm
            n2, d2 = cv2.GaussianBlur(WR, (0, 0), 2.0), cv2.GaussianBlur(Wm, (0, 0), 2.0)
            n9, d9 = cv2.GaussianBlur(WR, (0, 0), 9.0), cv2.GaussianBlur(Wm, (0, 0), 9.0)
            med = float(np.median(rc[valid]))
            coarse = (n9 + 0.05 * med) / (d9 + 0.05)
            a_fine = np.clip(d2 / 0.40, 0.0, 1.0)
            Gc[..., c] = a_fine * (n2 / np.maximum(d2, 1e-6)) + (1.0 - a_fine) * coarse
        Gt = np.clip(cv2.resize(Gc, (self.AW, self.AH), interpolation=cv2.INTER_LINEAR), 0.4, 2.2)
        self.Gs += 0.6 * (Gt - self.Gs)
        self.bg_small_adj = np.clip(self.bg_small_f * self.Gs + 0.5, 0, 255).astype(np.uint8)
        return True

    def _adj_job(self, G):
        try:
            Gf = cv2.resize(G, (self.PW, self.PH), interpolation=cv2.INTER_LINEAR)
            self.bg_adj = cv2.convertScaleAbs(cv2.multiply(self.bgf, Gf))   # swapped in atomically
            self.G_applied = G
        finally:
            self._adj_busy = False

    def _refresh_bg_full(self):
        """Rebuild the full-resolution clean plate in a worker thread when lighting moved enough to matter."""
        if self._adj_busy or float(np.max(np.abs(self.Gs - self.G_applied))) < 0.01:
            return
        self._adj_busy = True
        threading.Thread(target=self._adj_job, args=(self.Gs.copy(),), daemon=True).start()

    @staticmethod
    def _fill_holes(b):
        """Fill regions of 0s completely enclosed by 1s (clothes that match the background)."""
        h, w = b.shape
        pad = np.zeros((h + 2, w + 2), np.uint8)
        pad[1:-1, 1:-1] = b
        ff = pad.copy()
        cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
        holes = (ff == 0)
        return (b.astype(bool) | holes[1:-1, 1:-1]).astype(np.uint8)

    # ---- mask = model interior + clean-plate-difference edges
    def _build_mask(self, small):
        self.tick += 1
        w = self.worker
        if w is not None and not w.busy and self.tick % SEG_EVERY == 0:
            w.submit(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        prob = w.get_mask() if w is not None else None
        AW, AH = self.AW, self.AH
        if prob is not None and prob.shape != (AH, AW):
            prob = cv2.resize(prob, (AW, AH), interpolation=cv2.INTER_LINEAR)

        have_model = prob is not None and float((prob > 0.5).mean()) > 0.004
        if have_model:
            core = cv2.erode((prob > SEG_THRESH_CORE).astype(np.uint8), self.k_core)
            roi_b = cv2.dilate((prob > SEG_THRESH_ROI).astype(np.uint8), self.k_roi)
            self.far_ok = roi_b == 0
        else:
            core = None
            roi_b = None
            self.far_ok = None

        if self.tick % 3 == 0:
            self._fit_gain(small)

        d = cv2.absdiff(small, self.bg_small_adj)
        dm = np.maximum(np.maximum(d[..., 0], d[..., 1]), d[..., 2])
        dm = cv2.GaussianBlur(dm, (5, 5), 0)
        noise = float(np.median(dm[::2, ::2]))
        lo = min(max(3.5 * noise, DIFF_LO_MIN), DIFF_LO_MAX)
        diff = np.clip((dm.astype(np.float32) - lo) * (1.0 / DIFF_RAMP), 0.0, 1.0)
        matched = float((dm < lo).mean())
        if matched < MIN_MATCH_FRACTION:        # almost nothing resembles the saved background any more
            diff = np.zeros_like(diff)
            self.status = "BACKGROUND MISMATCH - press Ctrl+Alt+B (step out) to recapture"
            if not self._warned_scene:
                print("[!] " + self.status)
                self._warned_scene = True
        else:
            if matched > MIN_MATCH_FRACTION + 0.1:
                self._warned_scene = False
            self.status = ""

        if roi_b is not None:
            diff = diff * roi_b
        b = (diff > 0.5).astype(np.uint8)
        if core is not None:
            b |= core
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, self.k_edge) if core is None else b
        b = self._fill_holes(b)
        b = cv2.dilate(b, self.k_edge)
        m = b.astype(np.float32)

        if self.smooth is None:
            self.smooth = m
        else:
            self.smooth += MASK_SMOOTH_ALPHA * (m - self.smooth)
        cover = cv2.GaussianBlur(self.smooth, (5, 5), 0)
        self.last_cover = cover
        self.coverage = float(cover.mean())
        if not self.status and self.coverage < 0.002:
            self.status = "NOTHING DETECTED TO HIDE"
        return cover

    # ---- per-frame entry
    def process(self, frame, want_vanish):
        now = time.monotonic()
        dt = min(now - self.last_update_time, 0.1)   # a stall must not teleport the animation
        self.last_update_time = now

        target = 1.0 if (want_vanish and self.bg is not None) else 0.0
        dur = SNAP_SECONDS if self.mode == "snap" else FADE_SECONDS
        step = dt / max(dur, 0.01)
        if self.alpha < target:
            self.alpha = min(self.alpha + step, target)
        elif self.alpha > target:
            self.alpha = max(self.alpha - step, target)

        if self.bg is None or self.alpha <= 0.0:
            self.smooth = None
            self.far_ok = None
            if self.worker is not None:
                self.worker.clear()
            self.coverage = 0.0
            if self.bg is not None:                         # keep exposure match fresh while idle
                self.idle_tick += 1
                if self.idle_tick % 6 == 0:
                    small = cv2.resize(frame, (self.AW, self.AH), interpolation=cv2.INTER_AREA)
                    self._fit_gain(small)
                    self._refresh_bg_full()
            self.status = ""
            return frame

        small = cv2.resize(frame, (self.AW, self.AH), interpolation=cv2.INTER_AREA)
        cover = self._build_mask(small)
        if self.tick % 3 == 0:
            self._refresh_bg_full()
        p = np.float32(smoothstep(self.alpha))
        if self.mode == "snap":
            return self._snap_render(frame, small, cover, p)
        return self._fade_render(frame, cover, p)

    # ---- compositing helpers (work only inside the effect's bounding box)
    def _rect(self, *planes):
        reg = None
        for pl in planes:
            r = pl > 0.004
            reg = r if reg is None else (reg | r)
        x, y, w, h = cv2.boundingRect(reg.astype(np.uint8))
        if w == 0 or h == 0:
            return None
        x0, y0 = max(x - 1, 0), max(y - 1, 0)
        x1, y1 = min(x + w + 1, self.AW), min(y + h + 1, self.AH)
        return x0, y0, x1, y1

    def _up(self, plane, r, dtype_color=False):
        x0, y0, x1, y1 = r
        sw, sh = int(round((x1 - x0) * self.sx)), int(round((y1 - y0) * self.sy))
        return cv2.resize(np.ascontiguousarray(plane[y0:y1, x0:x1]), (sw, sh), interpolation=cv2.INTER_LINEAR)

    def _full_box(self, r):
        x0, y0, x1, y1 = r
        return (int(round(x0 * self.sx)), int(round(y0 * self.sy)),
                int(round(x1 * self.sx)), int(round(y1 * self.sy)))

    def _fade_render(self, frame, cover, p):
        a = cover * p
        r = self._rect(a)
        if r is None:
            return frame
        X0, Y0, X1, Y1 = self._full_box(r)
        af = np.ascontiguousarray(self._up(a, r))
        out = frame.copy()
        out[Y0:Y1, X0:X1] = cv2.blendLinear(frame[Y0:Y1, X0:X1], self.bg_adj[Y0:Y1, X0:X1], 1.0 - af, af)
        return out

    def _snap_render(self, frame, small, cover, p):
        if p >= 0.999:                                    # end state: pure replacement, no dust
            gone, dust_a, dust_c = cover, None, None
        else:
            xs, ys = self.X, self.Y
            D = np.clip((p * np.float32(1.0 + SPREAD) - self.front) * self.inv_spread, 0.0, 1.0)
            remain = np.clip((self.n_a + (np.float32(0.15) - np.float32(1.15) * D)) * np.float32(8.0), 0.0, 1.0)
            gone = cover * (1.0 - remain)
            travel = D * self.drift * self.noise_mid
            mapx = xs - travel
            mapy = ys + travel * np.float32(0.35) + (self.n_low - np.float32(0.5)) * travel * np.float32(0.5)
            self.g4[..., :3] = small
            self.g4[..., 3] = (cover * 255.0).astype(np.uint8)
            src = cv2.remap(self.g4, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
            env = np.clip(D * 5.0, 0.0, 1.0) * np.clip((1.0 - D) * 2.0, 0.0, 1.0)
            sparkle = np.clip((self.n_b - (0.25 + 0.45 * D)) * 10.0, 0.0, 1.0)
            dust_a = (src[..., 3].astype(np.float32) * np.float32(1.0 / 255.0)) * env * sparkle
            dust_c = np.clip(src[..., :3].astype(np.float32) * 0.85 + 15.0, 0, 255).astype(np.uint8)

        planes = [gone] if dust_a is None else [gone, dust_a]
        r = self._rect(*planes)
        if r is None:
            return frame
        X0, Y0, X1, Y1 = self._full_box(r)
        gf = np.ascontiguousarray(self._up(gone, r), dtype=np.float32)
        roi = cv2.blendLinear(frame[Y0:Y1, X0:X1], self.bg_adj[Y0:Y1, X0:X1], 1.0 - gf, gf)
        if dust_a is not None and float(dust_a.max()) > 0.003:
            x0, y0, x1, y1 = r
            dcol = cv2.resize(np.ascontiguousarray(dust_c[y0:y1, x0:x1]), (X1 - X0, Y1 - Y0),
                              interpolation=cv2.INTER_LINEAR)
            daf = np.ascontiguousarray(self._up(dust_a, r), dtype=np.float32)
            roi = cv2.blendLinear(roi, dcol, 1.0 - daf, daf)
        out = frame.copy()
        out[Y0:Y1, X0:X1] = roi
        return out

    def close(self):
        if self.worker is not None:
            self.worker.stop()


# ------------------------------------------------------------------ Global hotkeys (own thread -> never missed)
class GlobalHotkeys(threading.Thread):
    KEYS = {"bg": ord("B"), "snap": ord("V"), "effect": ord("E"), "settings": ord("C"), "arm": ord("S")}

    def __init__(self):
        super().__init__(daemon=True)
        self.ok = IS_WIN
        self.q = queue.Queue()
        self.running = True
        if self.ok:
            self.user32 = ctypes.windll.user32
            self.start()

    def _down(self, vk):
        return bool(self.user32.GetAsyncKeyState(vk) & 0x8000)

    def run(self):
        prev = {}
        while self.running:
            combo = self._down(0x11) and self._down(0x12)      # Ctrl + Alt
            for name, vk in self.KEYS.items():
                now = combo and self._down(vk)
                if now and not prev.get(name, False):
                    self.q.put(name)
                prev[name] = now
            time.sleep(0.01)

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def stop(self):
        self.running = False


# ------------------------------------------------------------------ Finger-snap detector (microphone)
class SnapDetector:
    """Detects a finger snap (or clap) from raw mono audio. Pure NumPy, device independent.

    A snap = sudden, very short, high-frequency burst far above the room's noise floor that dies away in
    well under 100 ms. Speech, music, 's' sounds and typing fail at least one of those tests.
    """
    HOP = 256

    def __init__(self, sr, sensitivity=5, cooldown=1.5):
        self.sr = sr
        self.k = 40.0 * (0.75 ** (sensitivity - 5))            # required peak over the noise floor (power ratio)
        self.cooldown_hops = int(cooldown * sr / self.HOP)
        n = 2 * self.HOP
        self.win = np.hanning(n).astype(np.float32)
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        self.hf = freqs >= 2000.0
        self.lo = (freqs >= 100.0) & (freqs < 1500.0)
        self.buf = np.zeros(n, np.float32)
        self.pending = np.zeros(0, np.float32)
        self.floor = None
        self.hist = [0.0, 0.0, 0.0]
        self.state = "idle"
        self.peak = 0.0
        self.wait = 0
        self.cool = 0
        self.level = 0.0                                         # for --snap-test meter (HF power / floor)

    def feed(self, samples):
        """Feed any number of mono float32 samples. Returns number of snaps detected in this chunk."""
        x = np.concatenate([self.pending, np.asarray(samples, np.float32).ravel()])
        n_hops = len(x) // self.HOP
        hits = 0
        for i in range(n_hops):
            hop = x[i * self.HOP:(i + 1) * self.HOP]
            self.buf = np.concatenate([self.buf[self.HOP:], hop])
            if self._step():
                hits += 1
        self.pending = x[n_hops * self.HOP:]
        return hits

    def _step(self):
        spec = np.abs(np.fft.rfft(self.buf * self.win)) ** 2
        e_hf = float(spec[self.hf].sum()) + 1e-12
        e_lo = float(spec[self.lo].sum()) + 1e-12
        if self.floor is None:
            self.floor = e_hf
        self.level = e_hf / max(self.floor, 1e-12)
        ratio = e_hf / (e_hf + e_lo)
        prev = (self.hist[0] + self.hist[1] + self.hist[2]) / 3.0
        fired = False
        if self.cool > 0:
            self.cool -= 1
        elif self.state == "idle":
            if e_hf > self.k * self.floor and e_hf > 6.0 * prev and ratio > 0.5 and e_hf > 1e-3:
                self.state, self.peak, self.wait = "verify", e_hf, 9
        else:                                                    # verify the burst dies away quickly
            self.peak = max(self.peak, e_hf)
            self.wait -= 1
            if self.wait <= 0:
                tail = (e_hf + self.hist[2] + self.hist[1]) / 3.0
                if tail < 0.15 * self.peak:
                    fired = True
                    self.cool = self.cooldown_hops
                self.state = "idle"
        if self.state == "idle" and e_hf < 3.0 * self.floor:
            self.floor = 0.985 * self.floor + 0.015 * e_hf         # adapt to the room only while quiet
        self.hist = [self.hist[1], self.hist[2], e_hf]
        return fired


class SnapListener:
    """Opens the microphone (needs: pip install sounddevice) and pushes 'snap' events to a queue."""

    def __init__(self, device=None, sensitivity=5, meter=False):
        import sounddevice as sd
        self.sd = sd
        info = sd.query_devices(device, "input")
        sr = int(info["default_samplerate"])
        self.det = SnapDetector(sr, sensitivity)
        self.q = queue.Queue()
        self.armed = True
        self.meter = meter
        self.stream = sd.InputStream(device=device, channels=1, samplerate=sr, blocksize=512,
                                     dtype="float32", callback=self._cb)
        self.stream.start()
        print(f"[+] Snap listener on: {info['name']} @ {sr} Hz  (sensitivity {sensitivity}/10)")

    def _cb(self, indata, frames, t, status):
        n = self.det.feed(indata[:, 0])
        if n and self.armed:
            self.q.put("snap")
        if self.meter:
            lvl = min(self.det.level, 400.0)
            print("\r[mic] " + "#" * int(lvl ** 0.5) + " " * (22 - int(lvl ** 0.5)) + (" SNAP!" if n else "      "), end="", flush=True)

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def stop(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


# ------------------------------------------------------------------ Virtual camera
VCAM_NAME = "Vanish Camera"          # DirectShow device name registered by the installer (Unity Capture, MIT licence)


def open_virtual_cam(w, h, fps, name=VCAM_NAME, log=print):
    """Bundled 'Vanish Camera' driver first (no OBS needed), then any other pyvirtualcam backend (e.g. OBS)."""
    try:
        import pyvirtualcam
    except Exception as e:
        log(f"[!] pyvirtualcam missing: {e!r}")
        return None
    last = None
    for backend, dev in (("unitycapture", name), ("unitycapture", None), (None, None)):
        try:
            kw = {}
            if backend:
                kw["backend"] = backend
            if dev:
                kw["device"] = dev
            cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR, **kw)
            log(f"[+] Virtual camera: {cam.device} via {getattr(cam, 'backend', '?')} ({w}x{h}@{fps})")
            return cam
        except Exception as e:
            last = e
    log(f"[!] Virtual camera unavailable: {last}")
    log("    Install the camera driver (the installer does this; or run 'Install camera driver.bat' as admin).")
    return None


# ------------------------------------------------------------------ Self test (no camera)
def selftest(w, h, use_model):
    print(f"[selftest] {w}x{h}  model={'on' if use_model else 'off'}")
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.stack([90 + 50 * xx / w, 110 + 40 * yy / h, 130 - 30 * xx / w], -1)
    bg = (bg + (cv2.GaussianBlur(rng.random((h, w, 3), dtype=np.float32), (0, 0), 25) - 0.5) * 120).clip(0, 255).astype(np.uint8)
    pm = np.zeros((h, w), np.uint8)
    cv2.ellipse(pm, (w // 2, int(h * 0.65)), (int(w * .18), int(h * .36)), 0, 0, 360, 255, -1)
    cv2.circle(pm, (w // 2, int(h * 0.28)), int(h * .13), 255, -1)
    person = np.empty_like(bg)
    person[:] = (60, 80, 200)
    live = bg.copy()
    live[pm > 0] = person[pm > 0]
    for mode in ("snap", "fade"):
        eng = VanishEngine(w, h, use_model=use_model)
        eng.mode = mode
        eng.set_background(bg)
        times, t_end = [], time.monotonic()
        clock = [t_end]
        real = time.monotonic
        time.monotonic = lambda: clock[0]                 # deterministic animation clock
        try:
            for i in range(110):
                clock[0] += 1 / 30
                t0 = time.perf_counter()
                out = eng.process(live, True)
                times.append((time.perf_counter() - t0) * 1000)
                time.sleep(0.004)
        finally:
            time.monotonic = real
        eng.close()
        err = np.abs(out.astype(np.float32) - bg.astype(np.float32)).mean(-1)[pm > 127]
        t = np.array(times[20:])
        print(f"  {mode.upper():5s} median {np.median(t):5.1f} ms  p95 {np.percentile(t, 95):5.1f} ms  "
              f"=> ~{1000 / max(np.median(t), 1e-3):4.0f} FPS capable | person-vs-background error {err.mean():4.1f} levels")
    print("  Target: median < 25 ms at 1280x720 (30 FPS). If higher, run with --res 960x540.")


def camtest(idx, w, h, fps):
    """Raw camera speed per backend/format with NO processing. Shows if freezes come from the driver."""
    print(f"[camtest] camera {idx}, {w}x{h} @ {fps}. Stay in view; ~4 s per row.")
    for name, api in BACKENDS:
        for mjpg in (True, False):
            cap = cv2.VideoCapture(idx, api)
            if not cap.isOpened():
                cap.release()
                print(f"  {name:6s} MJPG={mjpg!s:5s}  cannot open")
                continue
            if mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
            stamps, t_end = [], time.perf_counter() + 4.0
            while time.perf_counter() < t_end:
                ok, f = cap.read()
                if ok and f is not None:
                    stamps.append(time.perf_counter())
            cap.release()
            if len(stamps) < 5:
                print(f"  {name:6s} MJPG={mjpg!s:5s}  no frames")
                continue
            gaps = np.diff(stamps) * 1000
            print(f"  {name:6s} MJPG={mjpg!s:5s}  {len(stamps) / 4.0:5.1f} FPS | worst gap {gaps.max():5.0f} ms | "
                  f"freezes(>100ms): {int((gaps > 100).sum())}")
    print("  Pick the row with ~30 FPS and 0 freezes, then run: python vanish.py --backend <name>")


# ------------------------------------------------------------------ Main loop
def parse_args():
    ap = argparse.ArgumentParser(description="Vanish / Thanos snap virtual webcam effect")
    ap.add_argument("--cam", type=int, default=CAMERA_INDEX)
    ap.add_argument("--res", default=f"{PROC_W}x{PROC_H}", help="WxH, multiples of 20 (e.g. 960x540)")
    ap.add_argument("--fps", type=int, default=TARGET_FPS)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-model", action="store_true", help="background subtraction only (no MediaPipe)")
    ap.add_argument("--mode", choices=["snap", "fade"], default=EFFECT)
    ap.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto")
    ap.add_argument("--camtest", action="store_true", help="measure raw camera FPS/freezes per backend")
    ap.add_argument("--snap", action="store_true", help="vanish when you snap your fingers (microphone)")
    ap.add_argument("--snap-sens", type=int, default=5, choices=range(1, 11), help="1 = hard to trigger ... 10 = very sensitive")
    ap.add_argument("--mic", type=int, default=None, help="microphone device index")
    ap.add_argument("--snap-test", action="store_true", help="live microphone meter to tune --snap-sens")
    ap.add_argument("--selftest", action="store_true", help="benchmark + sanity check without a camera")
    return ap.parse_args()


class Pipeline:
    """Camera -> effect -> virtual camera. Used by both the command line and the control window."""

    def __init__(self, cam=0, w=PROC_W, h=PROC_H, fps=TARGET_FPS, backend="auto", use_model=True, mode=EFFECT,
                 snap=False, snap_sens=5, mic=None, vcam_name=VCAM_NAME, log=print, preview_cb=None, hotkeys=True):
        self.cam, self.W, self.H, self.fps, self.backend = cam, w, h, fps, backend
        self.use_model, self.mode0 = use_model, mode
        self.snap, self.snap_sens, self.mic = snap, snap_sens, mic
        self.vcam_name, self.log, self.preview_cb, self.use_hotkeys = vcam_name, log, preview_cb, hotkeys
        self.cmds = queue.Queue()
        self.stop_flag = threading.Event()
        self.info = dict(running=False, fps=0.0, ms=0.0, coverage=0.0, vanish=False, mode=mode, status="",
                         bg_ready=False, vcam="", snap_armed=False, snap_on=False, error="")
        self.grabber = self.engine = self.vcam = self.hot = self.snapper = None

    # ---- thread-safe control
    def send(self, action):
        self.cmds.put(action)

    def stop(self):
        self.stop_flag.set()

    def enable_snap(self, on, sens=None):
        """Turn finger-snap listening on/off while running (used by the control window)."""
        if sens is not None:
            self.snap_sens = sens
        if on and self.snapper is None:
            try:
                self.snapper = SnapListener(self.mic, self.snap_sens)
            except Exception as e:
                self.log(f"[!] Snap listening unavailable: {e!r}")
                return False
        if self.snapper is not None:
            self.snapper.armed = bool(on)
        return True

    def set_sens(self, sens):
        self.snap_sens = sens
        if self.snapper is not None:
            self.snapper.det.k = 40.0 * (0.75 ** (sens - 5))

    # ---- setup / teardown
    def open(self):
        cap = open_camera(self.cam, self.W, self.H, self.fps, self.backend)
        if cap is None:
            self.info["error"] = "No camera found. Close other apps using the webcam (Zoom, Teams, browser) or pick another camera."
            return False
        self.grabber = ThreadedGrabber(cap)
        self.grabber.start()
        frame, self.last_seq = self.grabber.wait_new(0, 3.0)
        if frame is None:
            self.info["error"] = "The camera produced no frames."
            self.grabber.stop()
            return False
        self.engine = VanishEngine(self.W, self.H, use_model=self.use_model)
        self.engine.mode = self.mode0
        if self.use_model and not self.engine.model_ok:
            self.log(f"[!] AI person model did not load ({self.engine.model_error}). Running in basic mode - results may be worse.")
        self.vcam = open_virtual_cam(self.W, self.H, self.fps, self.vcam_name, self.log)
        self.info["vcam"] = (self.vcam.device if self.vcam is not None else "")
        if self.use_hotkeys:
            self.hot = GlobalHotkeys()
        if self.snap:
            try:
                self.snapper = SnapListener(self.mic, self.snap_sens)
            except Exception as e:
                self.log(f"[!] Snap listening unavailable: {e!r}")
        return True

    def close(self):
        if self.hot is not None:
            self.hot.stop()
        if self.snapper is not None:
            self.snapper.stop()
        if self.grabber is not None:
            self.grabber.stop()
        if self.engine is not None:
            self.engine.close()
        if self.vcam is not None:
            try:
                self.vcam.close()
            except Exception:
                pass
        self.info["running"] = False

    # ---- command handling
    def _trigger(self, action):
        st, eng = self.state, self.engine
        if action == "bg":
            st["vanish"] = False
            st["countdown_end"] = time.monotonic() + 3.0
            self.acc = None
            self.log("[*] Background capture: step COMPLETELY out of frame and keep still.")
        elif action == "snap":
            if not eng.bg_ready:
                self.log("[!] Capture a background first.")
            else:
                st["vanish"] = not st["vanish"]
                self.log(f"[*] Vanish: {st['vanish']}")
        elif action == "effect":
            eng.mode = "fade" if eng.mode == "snap" else "snap"
            self.log(f"[*] Effect: {eng.mode.upper()}")
        elif action == "settings":
            self.grabber.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        elif action == "arm":
            if self.snapper is None:
                self.log("[!] Snap listening is off.")
            else:
                self.snapper.armed = not self.snapper.armed
                self.log(f"[*] Snap listening: {'ON' if self.snapper.armed else 'PAUSED'}")
        elif action == "mask":
            st["show_mask"] = not st["show_mask"]

    # ---- main loop (blocks until stop())
    def run(self):
        if self.engine is None and not self.open():
            self.log("[!] " + self.info["error"])
            return
        W, H, eng, info = self.W, self.H, self.engine, self.info
        self.state = {"vanish": False, "countdown_end": None, "settle": 0, "show_mask": False}
        self.acc, acc_n = None, 0
        info["running"] = True
        fps_t, n_frames, proc_ms = time.monotonic(), 0, 0.0
        last_seq = self.last_seq
        try:
            while not self.stop_flag.is_set():
                frame, seq = self.grabber.wait_new(last_seq)
                if frame is None or seq == last_seq:
                    continue
                last_seq = seq
                frame = fit_frame(frame, W, H)

                while True:
                    try:
                        self._trigger(self.cmds.get_nowait())
                    except queue.Empty:
                        break
                if self.hot is not None:
                    for act in self.hot.poll():
                        self._trigger(act)
                if self.snapper is not None:
                    for act in self.snapper.poll():
                        self._trigger(act)

                st, status = self.state, ""
                if st["countdown_end"] is not None:
                    rem = st["countdown_end"] - time.monotonic()
                    if rem > 0:
                        status = f"STEP OUT: {math.ceil(rem)}"
                    else:
                        st["countdown_end"] = None
                        st["settle"] = BG_SETTLE_FRAMES
                        self.acc = np.zeros((H, W, 3), np.float32)
                        acc_n = 0
                if self.acc is not None:
                    if st["settle"] > 0:
                        st["settle"] -= 1
                        status = "SETTLING..."
                    else:
                        cv2.accumulate(frame, self.acc)
                        acc_n += 1
                        status = f"CAPTURING BG {acc_n}/{BG_FRAMES}"
                        if acc_n >= BG_FRAMES:
                            eng.set_background((self.acc / acc_n).astype(np.uint8))
                            self.acc = None
                            status = ""
                            self.log("[+] Background saved. Step back in and vanish.")

                t0 = time.perf_counter()
                out = eng.process(frame, st["vanish"])
                proc_ms += (time.perf_counter() - t0) * 1000.0
                if self.vcam is not None:
                    self.vcam.send(out)
                n_frames += 1

                now = time.monotonic()
                if now - fps_t >= 2.0:
                    info["fps"], info["ms"] = n_frames / (now - fps_t), proc_ms / max(n_frames, 1)
                    self.log(f"[Stat] {info['fps']:4.1f} FPS | render {info['ms']:4.1f} ms | {eng.mode} | "
                             f"mask {eng.coverage * 100:4.1f}% of frame")
                    fps_t, n_frames, proc_ms = now, 0, 0.0
                info.update(coverage=eng.coverage, vanish=st["vanish"], mode=eng.mode, bg_ready=eng.bg_ready,
                            status=status or (eng.status if st["vanish"] else ""),
                            snap_on=self.snapper is not None, snap_armed=bool(self.snapper and self.snapper.armed))

                if self.preview_cb is not None and (seq % PREVIEW_EVERY == 0):
                    cover = eng.last_cover if st["show_mask"] else None
                    label = status or (eng.mode.upper() + " ON" if st["vanish"] else "VISIBLE")
                    key = self.preview_cb(out, label, cover, info["status"] if st["vanish"] else "")
                    if key == ord("q"):
                        break
                    elif key in (ord("b"), ord(" "), ord("e"), ord("s")):
                        self._trigger({ord("b"): "bg", ord(" "): "snap", ord("e"): "effect", ord("s"): "settings"}[key])
                    elif key == ord("m"):
                        st["show_mask"] = not st["show_mask"]
        except KeyboardInterrupt:
            pass
        finally:
            self.close()


def main():
    args = parse_args()
    W, H = (int(v) for v in args.res.lower().split("x"))
    assert W % 20 == 0 and H % 20 == 0, "--res must be multiples of 20"
    if args.selftest:
        selftest(W, H, not args.no_model)
        return
    if args.snap_test:
        try:
            lst = SnapListener(args.mic, args.snap_sens, meter=True)
        except Exception as e:
            print(f"[!] Microphone unavailable: {e!r}\n    Install with:  pip install sounddevice")
            return
        print("[snap-test] Snap your fingers. The bar shows loudness over room noise; SNAP! = detected. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        lst.stop()
        return
    if args.camtest:
        camtest(args.cam, W, H, args.fps)
        return
    apply_windows_optimizations()

    win = "Vanish Preview (q quit | space snap | b bg | e mode | m mask)"

    def cli_preview(out, label, cover, warn):
        if cover is not None:
            pv = cv2.cvtColor(cv2.resize((cover * 255).astype(np.uint8), (480, 270)), cv2.COLOR_GRAY2BGR)
        else:
            pv = cv2.resize(out, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(pv, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if warn:
            cv2.putText(pv, warn[:46], (10, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imshow(win, pv)
        return cv2.waitKey(1) & 0xFF

    pl = Pipeline(args.cam, W, H, args.fps, args.backend, not args.no_model, args.mode, args.snap, args.snap_sens,
                  args.mic, preview_cb=None if (args.no_preview or not PREVIEW) else cli_preview)
    if not pl.open():
        print("[!] " + pl.info["error"])
        end_windows_optimizations()
        return
    print("\n" + "=" * 64)
    print(" READY. Hotkeys: Ctrl+Alt+B bg | Ctrl+Alt+V snap | Ctrl+Alt+E effect | Ctrl+Alt+C cam | Ctrl+Alt+S snap-listen")
    print("=" * 64 + "\n")
    try:
        pl.run()
    finally:
        cv2.destroyAllWindows()
        end_windows_optimizations()


if __name__ == "__main__":
    main()
