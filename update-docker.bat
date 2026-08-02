@echo off
setlocal DisableDelayedExpansion
rem Generic name for new installs. Do not CALL: transferring control to the
rem compatibility entry point preserves its exact exit code and leaves only one
rem channel-validation/update implementation to maintain.
rem Re-quote each argument so a caller cannot splice cmd metacharacters into the
rem forwarded command line. An absent argument forwards as an empty quoted pair,
rem which the compatibility entry point reads back as no argument at all.
@"%~dp0update-docker-gpu.bat" "%~1" "%~2"
