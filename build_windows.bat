@echo off
setlocal

rem Builds dist\Daniel.exe: XDC1 with MSVC, then Daniel with PyInstaller.

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

call "%ROOT_DIR%build_xdc1.bat"
if errorlevel 1 exit /b 1
cd /d "%ROOT_DIR%"

set "VENV_DIR=%ROOT_DIR%.venv-windows-build"
if not exist "%VENV_DIR%\Scripts\python.exe" (
  python -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo Failed to create a virtualenv. Install Python 3 and rerun.
    exit /b 1
  )
)

"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip >nul
"%VENV_DIR%\Scripts\python.exe" -m pip install pyinstaller numpy pillow websocket-client
if errorlevel 1 exit /b 1

set "ADD_DATA=--add-data src\xdc1.exe;."
if exist src\icon.png set "ADD_DATA=%ADD_DATA% --add-data src\icon.png;."
if exist src\icon.ico set "ADD_DATA=%ADD_DATA% --add-data src\icon.ico;. --icon src\icon.ico"

"%VENV_DIR%\Scripts\python.exe" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --noconsole ^
  --name "Daniel" ^
  --collect-all numpy ^
  --collect-all PIL ^
  --hidden-import websocket ^
  %ADD_DATA% ^
  src\daniel.py
if errorlevel 1 exit /b 1

echo Build complete: dist\Daniel.exe
