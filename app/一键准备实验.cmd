@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt --timeout 120
if errorlevel 1 goto failed
if not exist "data\reviews.csv" (
  ".venv\Scripts\python.exe" prepare_data.py --movies 40
  if errorlevel 1 goto failed
)
if not exist "models\baseline.joblib" (
  ".venv\Scripts\python.exe" train.py
  if errorlevel 1 goto failed
)
if not exist "data\expanded_reviews.csv" (
  ".venv\Scripts\python.exe" prepare_data.py --expanded
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" train_enhanced.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 goto failed
echo 实验完成，请查看 reports\enhanced 文件夹，再运行“启动系统.cmd”。
pause
exit /b 0
:failed
echo 实验未完成，请查看上方错误。网络错误时可稍后重新运行。
pause
exit /b 1
