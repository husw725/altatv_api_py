from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import tempfile, os, requests, shutil
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

router = APIRouter()

# ------------------ 请求模型 ------------------
class SceneRequest(BaseModel):
    video_url: str
    threshold: float = 27.0


# ------------------ 下载视频 ------------------
def download_video(url: str, tmp_dir: str) -> str:
    os.makedirs(tmp_dir, exist_ok=True)
    filename = os.path.join(tmp_dir, "video.mp4")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"视频下载失败: {e}")
    return filename


# ------------------ 镜头检测 ------------------
def detect_scenes(video_path: str, threshold: float = 27.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    video_manager.release()

    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]


# ------------------ API ------------------
@router.post("/detect")
def scene_detect_api(req: SceneRequest):
    tmp_dir = tempfile.mkdtemp(prefix="scene_")
    try:
        video_path = download_video(req.video_url, tmp_dir)
        scenes = detect_scenes(video_path, req.threshold)
        print(f"Detected {len(scenes)} scenes.")
        result = [
            {"scene_id": i + 1, "start_sec": round(s[0], 3), "end_sec": round(s[1], 3)}
            for i, s in enumerate(scenes)
        ]
        return {"status": "success", "scene_count": len(scenes), "scenes": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ------------------ 调试入口 ------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="本地调试 SceneDetect")
    parser.add_argument("--video", required=True, help="本地视频路径")
    parser.add_argument("--threshold", type=float, default=27.0, help="镜头检测灵敏度")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] 文件不存在: {args.video}")
        exit(1)

    print(f"[DEBUG] 开始检测: {args.video} (threshold={args.threshold})")
    scenes = detect_scenes(args.video, args.threshold)

    result = [
        {"scene_id": i + 1, "start_sec": round(s[0], 3), "end_sec": round(s[1], 3)}
        for i, s in enumerate(scenes)
    ]
    print(json.dumps({"scene_count": len(scenes), "scenes": result}, indent=2, ensure_ascii=False))