import os
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg

def detect_scenes(video_path: str, output_dir: str, threshold: float = 30.0):
    """
    使用 PySceneDetect 检测视频中的场景并导出截图。
    Args:
        video_path (str): 视频路径
        output_dir (str): 输出场景帧目录
        threshold (float): 内容变化检测阈值
    Returns:
        List[Tuple[float, float]]: 每个场景的起止时间（秒）
    """
    os.makedirs(output_dir, exist_ok=True)

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()

    # 生成场景截图
    for i, (start_time, end_time) in enumerate(scene_list):
        time_str = f"{i:04d}_{start_time.get_seconds():.2f}-{end_time.get_seconds():.2f}"
        scene_frame_dir = os.path.join(output_dir, f"scene_{time_str}")
        os.makedirs(scene_frame_dir, exist_ok=True)
        split_video_ffmpeg([video_path], scene_list[i:i+1], scene_frame_dir, "jpg")

    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]