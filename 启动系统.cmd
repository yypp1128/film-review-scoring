@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "MACBERT_PYTHON=%USERPROFILE%\anaconda3\envs\llm\python.exe"
if exist "models\macbert\ratings.safetensors" if exist ".bert_web_dependencies\streamlit\__init__.py" if exist "%MACBERT_PYTHON%" goto macbert
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import streamlit, sklearn, pandas" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt --timeout 120
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
if errorlevel 1 goto failed
exit /b 0
:macbert
set "PYTHONPATH=%~dp0.bert_web_dependencies;%~dp0.bert_dependencies"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
set "HF_HUB_OFFLINE=1"
"%MACBERT_PYTHON%" -m streamlit run app.py --global.developmentMode false --server.address 127.0.0.1 --browser.gatherUsageStats false
if errorlevel 1 goto failed
exit /b 0
:failed
echo 启动失败，请保留窗口中的报错信息。
pause
exit /b 1
