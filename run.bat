@echo off
setlocal
if "%~1"=="" goto usage
if /i "%~1"=="install" goto install
if /i "%~1"=="jupyter" goto jupyter
if /i "%~1"=="lint" goto lint
if /i "%~1"=="format" goto format
if /i "%~1"=="test" goto test
if /i "%~1"=="clean" goto clean
goto usage

:install
poetry install
goto end

:jupyter
poetry run jupyter lab --notebook-dir=labs
goto end

:lint
poetry run ruff check labs/
if errorlevel 1 exit /b 1
poetry run black --check labs/
goto end

:format
poetry run black labs/
poetry run ruff check --fix labs/
goto end

:test
poetry run pytest -v
goto end

:clean
powershell -NoProfile -Command "Get-ChildItem -Path . -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^(__pycache__|\.pytest_cache|\.ipynb_checkpoints|\.ruff_cache)$' } | Remove-Item -Recurse -Force"
echo Clean done.
goto end

:usage
echo Usage: run.bat ^<command^>
echo.
echo Commands:
echo   install   - poetry install
echo   jupyter   - start JupyterLab
echo   lint      - ruff + black --check
echo   format    - black + ruff --fix
echo   test      - pytest
echo   clean     - remove __pycache__, .pytest_cache, etc.
exit /b 1

:end
endlocal
