@echo off
setlocal
chcp 65001 >nul
title Codex Cleanup
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Cleanup-Codex.ps1" %*
set "CLEANUP_CODE=%ERRORLEVEL%"
echo.
echo Exit code: %CLEANUP_CODE%
pause
exit /b %CLEANUP_CODE%
