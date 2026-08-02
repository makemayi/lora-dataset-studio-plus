@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Portable one-click updater for the Docker installation.
rem Double-click (or no argument) follows the latest published GitHub Release.
rem Pass "main" explicitly to opt into the preview channel.
set "LDS_UPDATE_REQUEST=%~1"
set "LDS_UPDATE_EXTRA=%~2"
setlocal EnableDelayedExpansion
if not "!LDS_UPDATE_EXTRA!"=="" (
    endlocal
    echo [ERROR] Too many arguments.
    echo Usage : update-docker-gpu.bat [stable^|main]
    pause
    exit /b 2
)
if "!LDS_UPDATE_REQUEST!"=="" (
    endlocal
    goto :channel_stable
)
if /I "!LDS_UPDATE_REQUEST!"=="stable" (
    endlocal
    goto :channel_stable
)
if /I "!LDS_UPDATE_REQUEST!"=="main" (
    endlocal
    goto :channel_main
)
endlocal
echo [ERROR] Unknown channel.
echo Usage : update-docker-gpu.bat [stable^|main]
pause
exit /b 2

:channel_main
set "LDS_UPDATE_CHANNEL=main"
goto :channel_ready

:channel_stable
set "LDS_UPDATE_CHANNEL=stable"

:channel_ready

pushd "%~dp0" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Cannot open the installation folder.
    pause
    exit /b 1
)

rem Keep the whole tail in one parsed block. The updater is then free to replace
rem this BAT while it is running without making cmd.exe read half of a new file.
(
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\update-docker-gpu.ps1" -Channel "%LDS_UPDATE_CHANNEL%" -InstallRoot "%~dp0."
    if errorlevel 1 (
        popd
        echo.
        echo The update did not complete. Your local data was preserved.
        pause
        exit /b 1
    )
    popd
    echo.
    echo Rebuild started. Check that the application becomes reachable.
    pause
    exit /b 0
)
