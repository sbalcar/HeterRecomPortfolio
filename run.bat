@echo off
set "curr_dir=%cd%"

cd /D "%~dp0"

cd src
python main.py

cd /d "%curr_dir%"
