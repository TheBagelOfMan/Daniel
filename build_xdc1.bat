@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  echo MSVC 2022 not found. Install the Visual Studio 2022 Build Tools.
  exit /b 1
)

call "%VCVARS%" >nul
if not exist "%ROOT_DIR%build\xdc1" mkdir "%ROOT_DIR%build\xdc1"

cd /d "%ROOT_DIR%xdc1"
cl /nologo /O2 /std:c++17 /EHsc /W4 /Fo"%ROOT_DIR%build\xdc1\\" /Fe"%ROOT_DIR%src\xdc1.exe" xdc1.cpp
if errorlevel 1 exit /b 1

echo Build complete: src\xdc1.exe
