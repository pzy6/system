@echo off
echo 启动银发守护者系统...

set PYTHONPATH=%PYTHONPATH%;%cd%\src

call conda activate silver_guardian

python src/main.py %*

pause