@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "MACBERT_PYTHON=%USERPROFILE%\anaconda3\envs\llm\python.exe"
if not exist "%MACBERT_PYTHON%" set "MACBERT_PYTHON=python"
set "PYTHONPATH=%~dp0.bert_dependencies"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_OFFLINE=1"
if "%~1"=="" (
    "%MACBERT_PYTHON%" score_macbert.py --interactive
    pause
    exit /b
)
"%MACBERT_PYTHON%" score_macbert.py %*
exit /b %errorlevel%
