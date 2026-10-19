# PyInstaller spec for Vanish Cam.   Build:  pyinstaller --noconfirm --clean VanishCam.spec
# Output: dist/VanishCam/VanishCam.exe          (the window friends double-click)
#         dist/VanishCam/VanishCam-console.exe  (diagnostics: --selftest / --camtest / --snap-test)
import os, sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_all

datas, binaries = [], []
hidden = ["tkinter", "_tkinter", "tkinter.ttk", "sounddevice", "mediapipe.python._framework_bindings",
          "mediapipe.python.solutions.selfie_segmentation"]
datas += collect_data_files("mediapipe")          # the .tflite / .binarypb model files the segmenter loads at runtime
binaries += collect_dynamic_libs("mediapipe")
for pkg in ("pyvirtualcam", "sounddevice", "_sounddevice_data"):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hidden += h
    except Exception as e:                         # optional packages must never break the build
        print("collect_all skipped", pkg, e)

# big packages mediapipe lists as dependencies but this app never uses
EXCLUDES = ["jax", "jaxlib", "scipy", "torch", "tensorflow", "cryptography", "pytest", "IPython", "jupyter",
            "notebook", "PyQt5", "PyQt6", "PySide2", "PySide6", "pandas", "sklearn"]

a = Analysis(["vanish_app.py"], pathex=["."], binaries=binaries, datas=datas, hiddenimports=hidden, excludes=EXCLUDES)
pyz = PYZ(a.pure)
WIN = sys.platform == "win32"
ICON = "assets/icon.ico" if WIN else None
GUI = WIN and os.environ.get("VANISH_NOCONSOLE", "1") == "1"      # VANISH_NOCONSOLE=0 -> console build (tests)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="VanishCam", console=not GUI, icon=ICON, upx=False)
exe_c = EXE(pyz, a.scripts, [], exclude_binaries=True, name="VanishCam-console", console=True, icon=ICON, upx=False)
coll = COLLECT(exe, exe_c, a.binaries, a.datas, strip=False, upx=False, name="VanishCam")
