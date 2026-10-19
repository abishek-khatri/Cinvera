; Inno Setup 6 script - builds installer_output\VanishCam-Setup.exe
[Setup]
AppId={{B3C6F0A2-7D44-4C7B-9C5E-3F7B1D2A9E10}
OutputDir=installer_output
AppName=Vanish Cam
AppVersion=1.0
AppPublisher=Abishek Khatri
DefaultDirName={autopf}\VanishCam
DefaultGroupName=Vanish Cam
OutputBaseFilename=VanishCam-Setup
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\VanishCam.exe
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"

[Files]
Source: "dist\VanishCam\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion
Source: "driver\*"; DestDir: "{app}\driver"; Flags: ignoreversion

[Icons]
Name: "{group}\Vanish Cam"; Filename: "{app}\VanishCam.exe"
Name: "{autodesktop}\Vanish Cam"; Filename: "{app}\VanishCam.exe"; Tasks: desktopicon

[Run]
Filename: "{sys}\regsvr32.exe"; Parameters: "/s ""{app}\driver\UnityCaptureFilter64.dll"" ""/i:UnityCaptureName=Vanish Camera"""; Flags: runhidden waituntilterminated; StatusMsg: "Installing the Vanish Camera virtual webcam..."
Filename: "{syswow64}\regsvr32.exe"; Parameters: "/s ""{app}\driver\UnityCaptureFilter32.dll"" ""/i:UnityCaptureName=Vanish Camera"""; Flags: runhidden waituntilterminated
Filename: "{app}\VanishCam.exe"; Description: "Start Vanish Cam"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\regsvr32.exe"; Parameters: "/s /u ""{app}\driver\UnityCaptureFilter64.dll"""; Flags: runhidden; RunOnceId: "UnregCam64"
Filename: "{syswow64}\regsvr32.exe"; Parameters: "/s /u ""{app}\driver\UnityCaptureFilter32.dll"""; Flags: runhidden; RunOnceId: "UnregCam32"
