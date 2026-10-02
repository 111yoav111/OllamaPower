@echo off
rem Start the OllamaPower helper agent: start_agent.bat --hub 192.168.1.10
rem Uses .venv if it exists, else ollamapower on PATH.
set "HERE=%~dp0.."
if exist "%HERE%\.venv\Scripts\ollamapower.exe" (
    "%HERE%\.venv\Scripts\ollamapower.exe" agent %*
) else (
    ollamapower agent %*
)
