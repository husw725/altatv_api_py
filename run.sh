#!/bin/bash
set -e

APP_NAME="video_scene_api"
PYTHON_BIN="python3"
VENV_DIR=".venv"

echo "==============================="
echo "🚀 Starting $APP_NAME on Ubuntu"
echo "==============================="

# 1️⃣ 创建虚拟环境（如果不存在）
if [ ! -d "$VENV_DIR" ]; then
  echo "📦 Creating virtual environment..."
  $PYTHON_BIN -m venv $VENV_DIR
fi

# 2️⃣ 激活虚拟环境
source $VENV_DIR/bin/activate

# 3️⃣ 安装依赖
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4️⃣ 启动 FastAPI 服务
echo "🌍 Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 5005 --reload