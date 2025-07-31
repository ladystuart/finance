@echo off
REM Move to script folder
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Updating pip and dependencies...
python.exe -m pip install --upgrade pip
pip install -r requirements.txt

echo Uninstalling keras (if exists)...
pip uninstall keras -y

echo Installing tf-keras...
pip install tf-keras

echo Executing main.py...
python main.py

pause
