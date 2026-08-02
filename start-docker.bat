@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "LAUNCH_ARGUMENT="
if "%~1"=="" goto :arguments_ready
if /i "%~1"=="--configure" set "LAUNCH_ARGUMENT=-Configure"
if /i "%~1"=="--rebuild" set "LAUNCH_ARGUMENT=-Rebuild"
if /i "%~1"=="--update-rebuild" set "LAUNCH_ARGUMENT=-UpdateRebuild"
if not defined LAUNCH_ARGUMENT goto :invalid_arguments
if not "%~2"=="" goto :invalid_arguments

:arguments_ready
pushd "%~dp0" >nul 2>&1
if errorlevel 1 exit /b 1
where.exe powershell.exe >nul 2>&1
if errorlevel 1 goto :powershell_missing
if not exist "%~dp0scripts\docker-launch.ps1" goto :launcher_missing

powershell.exe -NoLogo -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0scripts\docker-launch.ps1" -Stack studio %LAUNCH_ARGUMENT%
set "EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXIT_CODE%

:invalid_arguments
echo ERROR: Unknown launcher argument.
echo Supported options: --configure, --rebuild or --update-rebuild
exit /b 1

:powershell_missing
echo ERROR: Windows PowerShell 5.1 is required and was not found.
goto :failed

:launcher_missing
echo ERROR: scripts\docker-launch.ps1 is missing from the extracted project.
goto :failed

:failed
popd
endlocal & exit /b 1
