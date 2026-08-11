@echo off
chcp 65001 >nul
echo ========================================
echo  洗发水视频脚本项目 - 环境安装
echo ========================================
echo.

:: 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

:: 检查 CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())" 2>nul
if %errorlevel% neq 0 (
    echo [提示] 未检测到 CUDA，将使用 CPU 训练（极慢）
)

:: 创建虚拟环境
echo [1/3] 创建虚拟环境...
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat

:: 安装依赖
echo [2/3] 安装依赖...
pip install -r requirements.txt -q

echo [3/3] 安装完成！
echo.
echo 可用命令:
echo   python scripts/run_finetune.py    运行微调
echo   python scripts/run_inference.py   模型推理
echo   python scripts/run_app.py         启动 Web 界面
echo.
pause
