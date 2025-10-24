#!/bin/bash
set -e
set -e

# change working directory to the script's folder
cd "$(dirname "$0")"

APP_NAME="video_scene_api"
PYTHON_BIN="python3"
VENV_DIR=".venv"
GIT_BRANCH="master"

echo "==============================="
echo "🚀 Starting $APP_NAME on Ubuntu"
echo "==============================="

# 0️⃣ Pull latest code
echo "📦 Pulling latest code from Git..."
git fetch origin $GIT_BRANCH
git reset --hard origin/$GIT_BRANCH

# 1️⃣ Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "📚 Creating virtual environment..."
  $PYTHON_BIN -m venv $VENV_DIR
fi

# 2️⃣ Activate virtual environment
source $VENV_DIR/bin/activate

# 3️⃣ Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4️⃣ Start FastAPI server
echo "🌍 Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 5005 --reload