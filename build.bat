@echo off
setlocal
cd /d "%~dp0"
echo ===== Vanish Cam: build =====
if not exist .buildvenv\Scripts\python.exe (
  echo Creating build environment...
  py -3.12 -m venv .buildvenv 2>nul || py -3.11 -m venv .buildvenv 2>nul || py -3 -m venv .buildvenv
)
if not exist .buildvenv\Scripts\python.exe (
  echo Python not found. Install Python 3.11 or 3.12 from python.org ^(tick "Add to PATH"^) and run again.
  pause & exit /b 1
)
call .buildvenv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo pip install failed & pause & exit /b 1)
if not exist driver\UnityCaptureFilter64.dll (echo driver\UnityCaptureFilter64.dll is missing & pause & exit /b 1)
pyinstaller --noconfirm --clean VanishCam.spec || (echo PyInstaller failed & pause & exit /b 1)
xcopy /E /I /Y /Q driver dist\VanishCam\driver >nul
echo.
echo Portable app:  dist\VanishCam\VanishCam.exe
set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist %ISCC% (
  %ISCC% installer.iss && echo Installer for friends:  installer_output\VanishCam-Setup.exe
) else (
  echo Inno Setup 6 not found - installer skipped. Get it free from jrsoftware.org/isinfo.php and run build.bat again.
)
pause
