@echo off
setlocal EnableExtensions

rem Keep every relative Compose path anchored to this repository, even when the
rem launcher is double-clicked from a folder whose path contains spaces.
pushd "%~dp0" >nul 2>&1
if errorlevel 1 (
  echo ERROR: Windows could not open the LoRA Dataset Studio folder.
  echo Move the project to a folder you can access, then run this file again.
  pause
  exit /b 1
)

set "COMPOSE_FILE=docker-compose.gpu.yml"
set "CONTAINER_NAME=lora-dataset-studio-gpu"
set "APP_URL=http://127.0.0.1:5050/"
set "DOCKER_EXE="

rem Prefer Docker Desktop's normal per-user and machine-wide install locations.
rem PATH remains a fallback for valid custom installations, without letting an
rem unrelated docker.exe shadow an installed Docker Desktop CLI.
if not defined DOCKER_EXE if exist "%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe" set "DOCKER_EXE=%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe"
if not defined DOCKER_EXE if exist "%LOCALAPPDATA%\Programs\Docker\Docker\resources\bin\docker.exe" set "DOCKER_EXE=%LOCALAPPDATA%\Programs\Docker\Docker\resources\bin\docker.exe"
if not defined DOCKER_EXE if exist "%ProgramFiles%\Docker\Docker\resources\bin\docker.exe" set "DOCKER_EXE=%ProgramFiles%\Docker\Docker\resources\bin\docker.exe"
if not defined DOCKER_EXE if defined ProgramW6432 if exist "%ProgramW6432%\Docker\Docker\resources\bin\docker.exe" set "DOCKER_EXE=%ProgramW6432%\Docker\Docker\resources\bin\docker.exe"
for /f "delims=" %%D in ('where.exe docker.exe 2^>nul') do if not defined DOCKER_EXE set "DOCKER_EXE=%%~fD"

if not defined DOCKER_EXE goto :docker_not_found

rem BuildKit and registry authentication launch helper executables beside the
rem Docker CLI. Make that directory visible even when docker.exe was not on PATH.
for %%I in ("%DOCKER_EXE%") do set "PATH=%%~dpI;%PATH%"

"%DOCKER_EXE%" compose version >nul 2>&1
if errorlevel 1 goto :compose_not_found

rem Docker's CLI may be installed while the Docker Desktop engine is stopped.
"%DOCKER_EXE%" info >nul 2>&1
if not errorlevel 1 goto :engine_ready

set "DOCKER_DESKTOP="
if exist "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe" set "DOCKER_DESKTOP=%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
if not defined DOCKER_DESKTOP if exist "%LOCALAPPDATA%\Programs\Docker\Docker\Docker Desktop.exe" set "DOCKER_DESKTOP=%LOCALAPPDATA%\Programs\Docker\Docker\Docker Desktop.exe"
if not defined DOCKER_DESKTOP if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" set "DOCKER_DESKTOP=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
if not defined DOCKER_DESKTOP if defined ProgramW6432 if exist "%ProgramW6432%\Docker\Docker\Docker Desktop.exe" set "DOCKER_DESKTOP=%ProgramW6432%\Docker\Docker\Docker Desktop.exe"
if not defined DOCKER_DESKTOP goto :desktop_not_found

echo Docker is installed but is not running. Starting Docker Desktop...
start "" "%DOCKER_DESKTOP%" >nul 2>&1
if errorlevel 1 goto :desktop_start_failed

rem Wait at most three minutes for Docker Desktop's engine.
for /l %%I in (1,1,90) do (
  "%DOCKER_EXE%" info >nul 2>&1
  if not errorlevel 1 goto :engine_ready
  timeout /t 2 /nobreak >nul 2>&1
)
goto :engine_timeout

:engine_ready
if not exist "%COMPOSE_FILE%" goto :compose_file_missing

rem Copy the template once. An existing .env may contain private keys and user
rem choices, so this launcher never replaces or edits it.
if not exist ".env" (
  if not exist ".env.example" goto :env_example_missing
  copy ".env.example" ".env" >nul
  if errorlevel 1 goto :env_create_failed
  echo Created .env from .env.example. You can add API keys later in Settings.
)

rem This launcher deliberately owns an isolated, persistent ComfyUI installation
rem inside the repository. It never searches for or mounts another ComfyUI.
for %%D in ("run" "basedir" "data-docker-gpu" "bank-images") do (
  if not exist "%%~D\" mkdir "%%~D" >nul 2>&1
  if not exist "%%~D\" goto :directory_failed
)
set "LDS_COMFY_RUN=./run"
set "LDS_COMFY_BASEDIR=./basedir"
set "LDS_DATA=./data-docker-gpu"
set "LDS_BANK_SOURCES=./bank-images"
rem Docker Desktop exposes Windows bind mounts as root inside Linux containers.
rem Override Compose interpolation for this process only; never rewrite .env.
set "LDS_UID=0"
set "LDS_GID=0"
set "LDS_HOST_PORT=5050"
set "LDS_COMFY_HOST_PORT=8188"

echo Checking the Docker GPU configuration...
"%DOCKER_EXE%" compose -f "%COMPOSE_FILE%" config --quiet
if errorlevel 1 goto :config_invalid

echo Building and starting LoRA Dataset Studio. The first build can take a while...
"%DOCKER_EXE%" compose -f "%COMPOSE_FILE%" up -d --build
if errorlevel 1 goto :startup_failed

echo Waiting for LoRA Dataset Studio and ComfyUI to become healthy...
rem First boot installs a large GPU stack. Wait at most twenty-five minutes, while
rem still stopping immediately when Docker reports an unhealthy or exited unit.
for /l %%I in (1,1,300) do (
  call :check_container
  if errorlevel 2 goto :container_failed
  if not errorlevel 1 goto :container_healthy
  timeout /t 5 /nobreak >nul 2>&1
)
goto :health_timeout

:container_healthy
echo.
echo LoRA Dataset Studio is ready at %APP_URL%
start "" "%APP_URL%"
set "EXIT_CODE=0"
goto :finish

:check_container
"%DOCKER_EXE%" inspect --format "{{.State.Status}}" "%CONTAINER_NAME%" 2>nul | findstr /x /c:"exited" /c:"dead" /c:"restarting" >nul
if not errorlevel 1 exit /b 2
"%DOCKER_EXE%" inspect --format "{{.State.Health.Status}}" "%CONTAINER_NAME%" 2>nul | findstr /x /c:"unhealthy" >nul
if not errorlevel 1 exit /b 2
"%DOCKER_EXE%" inspect --format "{{.State.Health.Status}}" "%CONTAINER_NAME%" 2>nul | findstr /x /c:"healthy" >nul
if not errorlevel 1 exit /b 0
exit /b 1

:show_logs
echo.
echo Recent container logs:
"%DOCKER_EXE%" compose -f "%COMPOSE_FILE%" logs --tail 120
echo.
echo For more detail, run this command from the project folder:
echo   docker compose -f docker-compose.gpu.yml logs -f
exit /b 0

:docker_not_found
echo ERROR: Docker was not found.
echo Install Docker Desktop for Windows, then run this file again.
echo Download: https://www.docker.com/products/docker-desktop/
goto :failed

:compose_not_found
echo ERROR: Docker Compose is not available in this Docker installation.
echo Update Docker Desktop, make sure "docker compose version" works, then try again.
goto :failed

:desktop_not_found
echo ERROR: The Docker engine is not running and Docker Desktop could not be found.
echo Open Docker Desktop manually, wait until it says the engine is running, then try again.
goto :failed

:desktop_start_failed
echo ERROR: Windows could not start Docker Desktop.
echo Start Docker Desktop from the Start menu, wait until it is ready, then try again.
goto :failed

:engine_timeout
echo ERROR: Docker Desktop did not become ready within three minutes.
echo Check Docker Desktop for an error, restart it if needed, then run this file again.
goto :failed

:compose_file_missing
echo ERROR: %COMPOSE_FILE% is missing from the project folder.
echo Download or extract the complete LoRA Dataset Studio release, then try again.
goto :failed

:env_example_missing
echo ERROR: .env.example is missing, so a safe .env file could not be created.
echo Download or extract the complete release, then try again.
goto :failed

:env_create_failed
echo ERROR: Windows could not create .env in the project folder.
echo Check that the folder is writable, then try again.
goto :failed

:directory_failed
echo ERROR: Windows could not create the persistent Docker data folders.
echo Check that the project folder is writable and has enough free disk space.
goto :failed

:config_invalid
echo ERROR: Docker Compose rejected the GPU configuration.
echo Read the message above, correct any invalid value in .env, then try again.
goto :failed

:startup_failed
echo ERROR: Docker could not build or start the GPU container.
call :show_logs
echo Check that NVIDIA drivers and NVIDIA Container Toolkit support are available to Docker Desktop.
goto :failed

:container_failed
echo ERROR: The GPU container exited or became unhealthy during startup.
call :show_logs
echo Look for the first ERROR above. Check GPU support, disk space, and internet access, then try again.
goto :failed

:health_timeout
echo ERROR: The GPU container did not become healthy within twenty-five minutes.
call :show_logs
echo The first setup may still be installing dependencies or components. Check the logs and Docker Desktop, then try again.
goto :failed

:failed
set "EXIT_CODE=1"
echo.
echo LoRA Dataset Studio was not started. No containers or data were removed.
pause

:finish
popd
endlocal & exit /b %EXIT_CODE%
