"""Vanish Camera - control window (Windows). Wraps vanish.Pipeline; no command line needed."""
import base64
import json
import os
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

import vanish as V

APP_NAME = "Vanish Camera"
VERSION = "1.0"
DATA_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "VanishCam")
SETTINGS = os.path.join(DATA_DIR, "settings.json")
LOGFILE = os.path.join(DATA_DIR, "log.txt")


def base_dir():
    return os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))


def run_driver_script(name):
    """Run driver\\<name>.bat elevated (UAC prompt). Returns True if it was launched."""
    bat = os.path.join(base_dir(), "driver", name)
    if not (V.IS_WIN and os.path.exists(bat)):
        return False
    import ctypes
    rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", bat, None, os.path.dirname(bat), 1)
    return rc > 32


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} {VERSION}")
        self.resizable(False, False)
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            open(LOGFILE, "w").close()
        except Exception:
            pass
        self.cfg = dict(cam=0, mode="snap", snap=False, sens=5)
        try:
            self.cfg.update(json.load(open(SETTINGS)))
        except Exception:
            pass

        self.pl = None
        self.thread = None
        self.logq = queue.Queue()
        self.prev_lock = threading.Lock()
        self.prev = None
        self._photo = None
        self.opening = False
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(60, self.tick)

    # ------------------------------------------------------------- UI
    def _build(self):
        pad = dict(padx=10, pady=4)
        self.canvas = tk.Canvas(self, width=480, height=270, bg="#111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 4))
        self.canvas_text = self.canvas.create_text(240, 135, fill="#aaa", font=("Segoe UI", 12),
                                                   text="Press  Start camera  to begin")
        self.status = tk.StringVar(value="Camera is off")
        tk.Label(self, textvariable=self.status, font=("Segoe UI", 13, "bold"), anchor="w").grid(
            row=1, column=0, columnspan=3, sticky="we", **pad)
        self.hint = tk.StringVar(value="1) Start camera   2) Capture background (step out of frame)   3) Vanish!")
        tk.Label(self, textvariable=self.hint, fg="#555", anchor="w", justify="left", wraplength=470).grid(
            row=2, column=0, columnspan=3, sticky="we", padx=10)

        self.btn_start = ttk.Button(self, text="Start camera", command=self.toggle_start)
        self.btn_start.grid(row=3, column=0, sticky="we", **pad)
        ttk.Label(self, text="Camera #").grid(row=3, column=1, sticky="e")
        self.cam_var = tk.StringVar(value=str(self.cfg["cam"]))
        ttk.Combobox(self, textvariable=self.cam_var, values=["0", "1", "2", "3"], width=4, state="readonly").grid(
            row=3, column=2, sticky="w", **pad)

        self.btn_bg = ttk.Button(self, text="Capture background", command=lambda: self.cmd("bg"), state="disabled")
        self.btn_bg.grid(row=4, column=0, sticky="we", **pad)
        self.btn_vanish = ttk.Button(self, text="Vanish / Reappear", command=lambda: self.cmd("snap"), state="disabled")
        self.btn_vanish.grid(row=4, column=1, columnspan=2, sticky="we", **pad)

        self.mode_var = tk.StringVar(value=self.cfg["mode"])
        fr = ttk.Frame(self)
        fr.grid(row=5, column=0, columnspan=3, sticky="w", padx=10)
        ttk.Label(fr, text="Effect:").pack(side="left")
        for txt, val in (("Snap (dust)", "snap"), ("Fade", "fade")):
            ttk.Radiobutton(fr, text=txt, value=val, variable=self.mode_var, command=self.on_mode).pack(side="left", padx=6)

        self.snap_var = tk.BooleanVar(value=bool(self.cfg["snap"]))
        fr2 = ttk.Frame(self)
        fr2.grid(row=6, column=0, columnspan=3, sticky="we", padx=10, pady=2)
        ttk.Checkbutton(fr2, text="Vanish when I snap my fingers", variable=self.snap_var, command=self.on_snap).pack(side="left")
        ttk.Label(fr2, text="  Sensitivity").pack(side="left")
        self.sens_var = tk.IntVar(value=int(self.cfg["sens"]))
        ttk.Scale(fr2, from_=1, to=10, variable=self.sens_var, command=lambda v: self.on_sens(), length=110).pack(side="left")

        self.vcam_msg = tk.StringVar(value="")
        self.vcam_lbl = tk.Label(self, textvariable=self.vcam_msg, anchor="w", justify="left", wraplength=470)
        self.vcam_lbl.grid(row=7, column=0, columnspan=2, sticky="we", padx=10)
        self.btn_driver = ttk.Button(self, text="Install camera driver", command=self.install_driver)
        self.log_box = tk.Text(self, height=5, width=64, state="disabled", bg="#f6f6f6", font=("Consolas", 8))
        self.log_box.grid(row=9, column=0, columnspan=3, padx=10, pady=(4, 10))
        ttk.Label(self, text="Hotkeys (work inside Meet/Zoom too):  Ctrl+Alt+B background   Ctrl+Alt+V vanish   "
                             "Ctrl+Alt+E effect   Ctrl+Alt+S snap on/off", foreground="#555", wraplength=470).grid(
            row=8, column=0, columnspan=3, padx=10, sticky="w")

    # ------------------------------------------------------------- logging / settings
    def log(self, msg):
        self.logq.put(str(msg))
        try:
            with open(LOGFILE, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n")
        except Exception:
            pass

    def save_cfg(self):
        self.cfg.update(cam=int(self.cam_var.get()), mode=self.mode_var.get(), snap=bool(self.snap_var.get()),
                        sens=int(self.sens_var.get()))
        try:
            json.dump(self.cfg, open(SETTINGS, "w"))
        except Exception:
            pass

    # ------------------------------------------------------------- pipeline control
    def toggle_start(self):
        if self.pl is None and not self.opening:
            self.start()
        else:
            self.stop()

    def start(self):
        self.save_cfg()
        self.opening = True
        self.btn_start.config(text="Starting...", state="disabled")
        self.status.set("Starting camera...")
        self.pl = V.Pipeline(cam=self.cfg["cam"], mode=self.cfg["mode"], snap=self.cfg["snap"],
                             snap_sens=self.cfg["sens"], log=self.log, preview_cb=self.on_preview)
        self.thread = threading.Thread(target=self._worker, args=(self.pl,), daemon=True)
        self.thread.start()

    def _worker(self, pl):
        try:
            if not pl.open():
                self.log("[!] " + pl.info["error"])
                pl.info["failed"] = pl.info["error"]
                return
            pl.info["opened"] = True
            pl.run()
        except Exception:
            tb = traceback.format_exc()
            self.log("[!] Crash:\n" + tb)
            pl.info["failed"] = "The app hit an error - see the log below."
        finally:
            pl.info["finished"] = True

    def stop(self):
        pl, th = self.pl, self.thread
        if pl is not None:
            pl.stop()
        if th is not None:
            th.join(timeout=4)
        self.pl = self.thread = None
        self.opening = False
        self.btn_start.config(text="Start camera", state="normal")
        for b in (self.btn_bg, self.btn_vanish):
            b.config(state="disabled")
        self.status.set("Camera is off")
        self.canvas.delete("img")
        self.canvas.itemconfig(self.canvas_text, state="normal")

    def cmd(self, action):
        if self.pl is not None:
            self.pl.send(action)

    def on_preview(self, out, label, cover, warn):          # runs in the pipeline thread
        small = cv2.resize((cover * 255).astype(np.uint8) if cover is not None else out, (480, 270),
                           interpolation=cv2.INTER_AREA)
        if small.ndim == 2:
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        with self.prev_lock:
            self.prev = small
        return None

    def on_mode(self):
        if self.pl is not None and self.pl.engine is not None:
            self.pl.engine.mode = self.mode_var.get()
        self.save_cfg()

    def on_snap(self):
        if self.pl is not None:
            self.pl.enable_snap(self.snap_var.get(), int(self.sens_var.get()))
        self.save_cfg()

    def on_sens(self):
        if self.pl is not None:
            self.pl.set_sens(int(round(float(self.sens_var.get()))))
        self.save_cfg()

    def install_driver(self):
        if run_driver_script("install_driver.bat"):
            self.log("[*] Camera driver installer started. Accept the Windows prompt, then press Stop and Start camera again.")
        else:
            self.log("[!] Could not start the driver installer (driver folder missing).")

    # ------------------------------------------------------------- periodic UI refresh (main thread)
    def tick(self):
        try:
            while True:
                line = self.logq.get_nowait()
                self.log_box.config(state="normal")
                self.log_box.insert("end", line + "\n")
                self.log_box.see("end")
                self.log_box.config(state="disabled")
        except queue.Empty:
            pass

        pl = self.pl
        if pl is not None:
            info = pl.info
            if info.get("failed"):
                self.status.set(info["failed"])
                self.stop()
            elif info.get("opened"):
                if self.opening:
                    self.opening = False
                    self.btn_start.config(text="Stop camera", state="normal")
                    for b in (self.btn_bg, self.btn_vanish):
                        b.config(state="normal")
                    if info.get("vcam"):
                        self.vcam_msg.set(f"In Meet / Zoom / Teams choose the camera:  {V.VCAM_NAME}")
                        self.vcam_lbl.config(fg="#0a6b0a")
                        self.btn_driver.grid_remove()
                    else:
                        self.vcam_msg.set("Camera driver not installed - other apps can't see this yet.")
                        self.vcam_lbl.config(fg="#b00020")
                        self.btn_driver.grid(row=7, column=2, padx=10)
                    if self.cfg["snap"] is False:
                        pass
                st = info.get("status") or ""
                if st:
                    self.status.set(st)
                elif not info["bg_ready"]:
                    self.status.set("Step 2: press 'Capture background' and step out of frame")
                else:
                    self.status.set(("VANISHED  " if info["vanish"] else "VISIBLE  ") +
                                    f"({info['fps']:.0f} fps, {info['ms']:.0f} ms)")
            with self.prev_lock:
                img = self.prev
                self.prev = None
            if img is not None:
                ok, png = cv2.imencode(".png", img)
                if ok:
                    self._photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
                    self.canvas.delete("img")
                    self.canvas.create_image(0, 0, anchor="nw", image=self._photo, tags="img")
                    self.canvas.itemconfig(self.canvas_text, state="hidden")
        self.after(60, self.tick)

    def on_close(self):
        self.save_cfg()
        self.stop()
        self.destroy()


def main():
    if len(sys.argv) > 1:                       # VanishCam.exe --selftest / --camtest / --snap-test keep working
        V.main()
        return
    if V.IS_WIN:
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        V.apply_windows_optimizations()
    try:
        App().mainloop()
    finally:
        V.end_windows_optimizations()


if __name__ == "__main__":
    main()
