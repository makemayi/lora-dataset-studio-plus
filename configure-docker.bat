@echo off
setlocal EnableExtensions DisableDelayedExpansion
call "%~dp0start-docker.bat" --configure
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
