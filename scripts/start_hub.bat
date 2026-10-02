@echo off
rem Start the OllamaPower hub. Uses .venv if it exists, else ollamapower on PATH.
rem Extra arguments go to "ollamapower hub", e.g. --no-tui.
set "HERE=%~dp0.."
if exist "%HERE%\.venv\Scripts\ollamapower.exe" (
    "%HERE%\.venv\Scripts\ollamapower.exe" hub %*
) else (
    ollamapower hub %*
)
