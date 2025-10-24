#!/bin/bash
set -e

# 切换到脚本所在目录
cd "$(dirname "$0")"

APP_NAME="video_scene_api"
PYTHON_BIN="python3"
VENV_DIR=".venv"
GIT_BRANCH="main"

echo "==============================="
echo "🚀 Starting $APP_NAME on Ubuntu"
echo "==============================="

# 0️⃣ 拉取最新代码 (fail-safe)
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "📦 Pulling latest code from Git..."
    git pull origin $GIT_BRANCH || echo "⚠️ Git pull failed, continuing..."
else
    echo "⚠️ Not a git repository, skipping git pull."
fi

# 1️⃣ 创建虚拟环境（如果不存在）
if [ ! -d "$VENV_DIR" ]; then
  echo "📚 Creating virtual environment..."
  $PYTHON_BIN -m venv $VENV_DIR
fi

# 2️⃣ 激活虚拟环境
source $VENV_DIR/bin/activate

# 3️⃣ 安装依赖
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4️⃣ 启动 FastAPI 服务
echo "🌍 Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 5005 --reload