import os
import numpy as np
from collections import defaultdict
from utils.face_utils import extract_feature, cosine_sim, get_role_label
import cv2

SIM_THRESHOLD = 0.55
DET_THRESHOLD = 0.65


def normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalize feature"""
    return v / (np.linalg.norm(v) + 1e-6)


def compute_clarity(img_path: str) -> float:
    """
    返回 0~1，值越大图越清晰
    用 Laplacian variance 判断图像清晰度
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.3  # 兜底，避免异常影响聚类

    lap = cv2.Laplacian(img, cv2.CV_64F)
    score = lap.var()

    # 归一化至 [0, 1]
    score = min(score / 1500.0, 1.0)
    return max(score, 0.05)    # 低清晰度图最少也给 0.05 权重


def group_roles(
    input_dir: str,
    det_threshold: float = DET_THRESHOLD,
    sim_threshold: float = SIM_THRESHOLD
) -> dict:
    """
    按人脸特征聚类分组（加入清晰度加权 + 多样本中心更新）
    """
    role_centroids = {}                    # 角色中心特征
    role_feature_list = defaultdict(list)  # 存储 (feat, weight)
    role_images = defaultdict(set)         # 使用 set 防止重复
    next_role_id = 0

    file_list = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for img_name in file_list:
        img_path = os.path.join(input_dir, img_name)
        clarity = compute_clarity(img_path)

        # 提取特征（可能多张脸）
        features = extract_feature(img_path, det_threshold=det_threshold)
        if not features:
            role_images["other"].add(img_name)
            continue

        for feat in features:
            feat = normalize(np.array(feat))

            matched_role = None
            best_sim = -1

            # 1) 与所有角色中心比较
            for role, centroid in role_centroids.items():
                sim = cosine_sim(feat, centroid)
                if sim > best_sim:
                    best_sim = sim
                    matched_role = role

            # 2) 不足阈值 → 新角色
            if matched_role is None or best_sim < sim_threshold:
                new_role = get_role_label(next_role_id)
                next_role_id += 1

                role_feature_list[new_role].append((feat, clarity))
                role_centroids[new_role] = feat     # 首张图直接作为中心
                role_images[new_role].add(img_name)
                continue

            # 3) 匹配到已有角色 → 使用清晰度权重更新
            role_feature_list[matched_role].append((feat, clarity))

            features_arr = np.array([f for f, w in role_feature_list[matched_role]])
            weights_arr = np.array([w for f, w in role_feature_list[matched_role]])

            # 加权更新 centroid
            new_centroid = np.average(features_arr, axis=0, weights=weights_arr)
            role_centroids[matched_role] = normalize(new_centroid)

            role_images[matched_role].add(img_name)

    return {k: list(v) for k, v in role_images.items()}