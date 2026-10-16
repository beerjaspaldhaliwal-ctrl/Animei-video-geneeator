; Anime Studio - Windows Setup (NSIS 3).
; Build on Linux or Windows with:  installer/build_installer.sh   (needs makensis)
; SRC (folder with the app files) and OUTFILE are passed by the build script.

Unicode true
!define APPNAME "Anime Studio"
!define APPVERSION "1.2.0"
!define REGKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\AnimeStudio"
!ifndef SRC
  !error "Run installer/build_installer.sh (SRC is not defined)"
!endif
!ifndef OUTFILE
  !define OUTFILE "AnimeStudio-Setup.exe"
!endif

Name "${APPNAME}"
Caption "${APPNAME} ${APPVERSION} Setup"
BrandingText "${APPNAME} ${APPVERSION}"
OutFile "${OUTFILE}"
; Not "Program Files": the app saves projects and AI models next to itself.
InstallDir "C:\AnimeStudio"
InstallDirRegKey HKCU "${REGKEY}" "InstallLocation"
RequestExecutionLevel user
SetCompressor /SOLID lzma
ShowInstDetails show

VIProductVersion "${APPVERSION}.0"
VIAddVersionKey "ProductName" "${APPNAME}"
VIAddVersionKey "FileDescription" "${APPNAME} Setup"
VIAddVersionKey "FileVersion" "${APPVERSION}"
VIAddVersionKey "ProductVersion" "${APPVERSION}"
VIAddVersionKey "LegalCopyright" "Free software"

!include "MUI2.nsh"
!include "nsDialogs.nsh"
!include "LogicLib.nsh"
!include "StrFunc.nsh"
${StrStr}

Var GpuChoice
Var RadioAmd7
Var RadioAmd9
Var RadioNvidia
Var RadioCpu

!define MUI_ICON "anime_studio.ico"
!define MUI_UNICON "anime_studio.ico"
!define MUI_ABORTWARNING

!define MUI_WELCOMEPAGE_TITLE "Welcome to ${APPNAME}"
!define MUI_WELCOMEPAGE_TEXT "Make your own anime series, free, on your own PC.$\r$\n$\r$\nSetup copies ${APPNAME} to your computer, then downloads what it needs: a private copy of Python and PyTorch for your graphics card (a few GB). Keep your internet connected.$\r$\n$\r$\nThe AI models themselves (about 15 GB) download later, the first time you use each feature.$\r$\n$\r$\nTip: update your graphics driver before you start."
!insertmacro MUI_PAGE_WELCOME

!define MUI_DIRECTORYPAGE_TEXT_TOP "Choose a folder on a drive with at least 30 GB free (the AI models are big). Don't pick 'Program Files': ${APPNAME} saves your anime projects in this folder."
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE DirectoryLeave
!insertmacro MUI_PAGE_DIRECTORY

Page custom GpuPage GpuPageLeave

!insertmacro MUI_PAGE_INSTFILES

!define MUI_FINISHPAGE_TITLE "${APPNAME} is ready"
!define MUI_FINISHPAGE_TEXT "Start it any time with the ${APPNAME} icon on your desktop.$\r$\n$\r$\nFirst time? Open the Make Episode tab, tick Draft mode and click Make episode to try the example series."
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Start ${APPNAME} now"
!define MUI_FINISHPAGE_RUN_FUNCTION StartApp
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"


Function .onInit
  StrCpy $GpuChoice "1"
FunctionEnd

Function DirectoryLeave
  ${StrStr} $0 "$INSTDIR" "Program Files"
  ${If} $0 != ""
    MessageBox MB_ICONEXCLAMATION|MB_OK "Please choose a folder outside 'Program Files', for example C:\AnimeStudio."
    Abort
  ${EndIf}
FunctionEnd

Function GpuPage
  !insertmacro MUI_HEADER_TEXT "Graphics card" "${APPNAME} installs the AI engine built for your graphics card."
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 20u "Which graphics card does this PC have? (Not sure? Open Task Manager > Performance > GPU.)"
  Pop $0
  ${NSD_CreateRadioButton} 8u 26u 100% 12u "AMD Radeon RX 7000 series  (RX 7600, RX 7600 XT, 7700 XT, 7800 XT, 7900 ...)"
  Pop $RadioAmd7
  ${NSD_CreateRadioButton} 8u 44u 100% 12u "AMD Radeon RX 9000 series  (RX 9060 XT, 9070, 9070 XT)"
  Pop $RadioAmd9
  ${NSD_CreateRadioButton} 8u 62u 100% 12u "NVIDIA GeForce RTX  (20, 30, 40 or 50 series)"
  Pop $RadioNvidia
  ${NSD_CreateRadioButton} 8u 80u 100% 12u "No graphics card  (works, but very slow)"
  Pop $RadioCpu
  ${NSD_CreateLabel} 0 104u 100% 30u "Ryzen 7000/9000 processors also have small built-in graphics. That's fine: ${APPNAME} automatically uses your real graphics card."
  Pop $0
  ${Select} $GpuChoice
    ${Case} "2"
      ${NSD_Check} $RadioAmd9
    ${Case} "3"
      ${NSD_Check} $RadioNvidia
    ${Case} "4"
      ${NSD_Check} $RadioCpu
    ${CaseElse}
      ${NSD_Check} $RadioAmd7
  ${EndSelect}
  nsDialogs::Show
FunctionEnd

Function GpuPageLeave
  StrCpy $GpuChoice "1"
  ${NSD_GetState} $RadioAmd9 $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $GpuChoice "2"
  ${EndIf}
  ${NSD_GetState} $RadioNvidia $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $GpuChoice "3"
  ${EndIf}
  ${NSD_GetState} $RadioCpu $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $GpuChoice "4"
  ${EndIf}
FunctionEnd

Function StartApp
  ExecShell "open" "$INSTDIR\run_windows.bat"
FunctionEnd


Section "Anime Studio" SecMain
  SetOutPath "$INSTDIR"
  File /r "${SRC}\*.*"
  File "anime_studio.ico"
  CreateDirectory "$INSTDIR\projects"
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; Shortcuts (they start in the install folder)
  CreateShortCut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\run_windows.bat" "" "$INSTDIR\anime_studio.ico" 0
  CreateDirectory "$SMPROGRAMS\${APPNAME}"
  CreateShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\run_windows.bat" "" "$INSTDIR\anime_studio.ico" 0
  CreateShortCut "$SMPROGRAMS\${APPNAME}\My anime projects.lnk" "$INSTDIR\projects"
  CreateShortCut "$SMPROGRAMS\${APPNAME}\Repair or update ${APPNAME}.lnk" "$INSTDIR\install_windows.bat" "" "$INSTDIR\anime_studio.ico" 0
  CreateShortCut "$SMPROGRAMS\${APPNAME}\Uninstall ${APPNAME}.lnk" "$INSTDIR\Uninstall.exe"

  ; "Apps & features" entry (current user only, no admin needed)
  WriteRegStr HKCU "${REGKEY}" "DisplayName" "${APPNAME}"
  WriteRegStr HKCU "${REGKEY}" "DisplayVersion" "${APPVERSION}"
  WriteRegStr HKCU "${REGKEY}" "DisplayIcon" "$INSTDIR\anime_studio.ico"
  WriteRegStr HKCU "${REGKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${REGKEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKCU "${REGKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${REGKEY}" "NoRepair" 1

  ; Download Python, PyTorch and the libraries in a visible window.
  DetailPrint "Downloading Python, PyTorch and libraries in the black window. This can take 10-30 minutes..."
  ExecWait '"$SYSDIR\cmd.exe" /c ""$INSTDIR\install_windows.bat" $GpuChoice"' $0
  ${If} $0 != 0
    DetailPrint "Some downloads failed (code $0)."
    MessageBox MB_ICONEXCLAMATION|MB_OK "Some downloads didn't finish. Check your internet connection, then open Start menu > ${APPNAME} > Repair or update ${APPNAME}."
  ${Else}
    DetailPrint "All done."
  ${EndIf}
SectionEnd


Section "Uninstall"
  Delete "$DESKTOP\${APPNAME}.lnk"
  RMDir /r "$SMPROGRAMS\${APPNAME}"
  DeleteRegKey HKCU "${REGKEY}"

  ; Safety: only delete things if this really is an Anime Studio folder.
  ${IfNot} ${FileExists} "$INSTDIR\app.py"
    Goto done
  ${EndIf}
  RMDir /r "$INSTDIR\venv"
  RMDir /r "$INSTDIR\tools"
  RMDir /r "$INSTDIR\models"
  RMDir /r "$INSTDIR\anime_studio"
  RMDir /r "$INSTDIR\.gradio"
  Delete "$INSTDIR\app.py"
  Delete "$INSTDIR\requirements.txt"
  Delete "$INSTDIR\constraints.txt"
  Delete "$INSTDIR\settings.json"
  Delete "$INSTDIR\install_windows.bat"
  Delete "$INSTDIR\run_windows.bat"
  Delete "$INSTDIR\README.md"
  Delete "$INSTDIR\anime_studio.ico"
  MessageBox MB_YESNO|MB_ICONQUESTION|MB_DEFBUTTON2 "Also delete your anime projects (scripts, characters, finished episodes) and LoRA files?$\r$\n$\r$\nChoose No to keep them in:$\r$\n$INSTDIR\projects" /SD IDNO IDNO keep_projects
    RMDir /r "$INSTDIR\projects"
    RMDir /r "$INSTDIR\loras"
  keep_projects:
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  done:
SectionEnd
