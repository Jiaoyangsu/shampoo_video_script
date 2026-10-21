@echo off
chcp 65001 >nul
echo ========================================
echo  洗发水视频脚本项目 - 环境安装
echo ========================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if %errorlevel% neq 0 (
    echo [错误] 需要 Python 3.10+（Gradio 5+）
    pause
    exit /b 1
)

if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat

echo [1/2] 安装核心依赖（数据清洗 / 评测 / UI）...
pip install -r requirements-dev.txt -q

echo [2/2] 完成。GPU 训练请在服务器上额外安装：
echo   pip install -r requirements-train.txt
echo.
echo 可用命令:
echo   python scripts/clean_data.py
echo   python scripts/run_eval.py --backend mock --limit 3 --out-dir evaluation/results/mock
echo   python scripts/run_app.py
echo.
pause
