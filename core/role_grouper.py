import os
import shutil
from collections import defaultdict
from utils.face_utils import extract_feature, cosine_sim, get_role_label

SIM_THRESHOLD = 0.6  # 两个人脸特征相似度阈值

def group_roles(input_dir: str, 
                output_dir: str, 
                det_threshold: float = 0.75,
                sim_threshold: float = SIM_THRESHOLD) -> dict:
    """
    按人脸特征聚类分组
    Args:
        input_dir: 本地图片目录
        output_dir: 分组输出目录
        det_threshold: 人脸检测置信度阈值
        sim_threshold: 特征相似度阈值
    Returns:
        dict: {角色名: [图片文件名列表]}
    """
    role_features = {}  # 记录每个角色的参考特征
    role_images = defaultdict(list)
    next_role_id = 0

    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # 提取图片中所有人脸特征
        features = extract_feature(img_path, det_threshold=det_threshold)
        if not features:
            # 没检测到人脸的图片归类到 "other"
            role_images["other"].append(img_name)
            continue

        for feat in features:
            matched_role = None
            best_sim = 0
            # 和已有角色特征比较
            for role, ref_feat in role_features.items():
                sim = cosine_sim(feat, ref_feat)
                if sim > best_sim:
                    best_sim = sim
                    matched_role = role

            if matched_role and best_sim >= sim_threshold:
                # 匹配成功
                role_images[matched_role].append(img_name)
            else:
                # 新角色
                new_role = get_role_label(next_role_id)
                role_features[new_role] = feat
                role_images[new_role].append(img_name)
                next_role_id += 1

    # 保存图片到分组目录
    for role, images in role_images.items():
        role_dir = os.path.join(output_dir, f"role_{role}")
        os.makedirs(role_dir, exist_ok=True)
        for img_name in set(images):
            src = os.path.join(input_dir, img_name)
            dst = os.path.join(role_dir, img_name)
            if not os.path.exists(dst):
                shutil.copy(src, dst)

    return dict(role_images)