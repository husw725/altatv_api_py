import os
import numpy as np
from collections import defaultdict
from utils.face_utils import extract_feature, cosine_sim, get_role_label

SIM_THRESHOLD = 0.55
DET_THRESHOLD = 0.65

def normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalize feature"""
    return v / (np.linalg.norm(v) + 1e-6)


def group_roles(
    input_dir: str,
    det_threshold: float = DET_THRESHOLD,
    sim_threshold: float = SIM_THRESHOLD
) -> dict:
    """
    按人脸特征聚类分组（支持均值向量 + 多样本更新）
    """
    role_centroids = {}          # 角色中心特征
    role_feature_list = defaultdict(list)  # 角色全部特征，用于更新中心
    role_images = defaultdict(set)         # 使用 set 防止重复
    next_role_id = 0

    file_list = [f for f in os.listdir(input_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img_name in file_list:
        img_path = os.path.join(input_dir, img_name)

        # 提取特征（可能多张脸）
        features = extract_feature(img_path, det_threshold=det_threshold)
        if not features:
            role_images["other"].add(img_name)
            continue

        for feat in features:
            feat = normalize(feat)

            matched_role = None
            best_sim = -1

            # 1) 与所有已知角色中心比较
            for role, centroid in role_centroids.items():
                sim = cosine_sim(feat, centroid)
                if sim > best_sim:
                    best_sim = sim
                    matched_role = role

            # 2) 阈值内 → 新角色
            if matched_role is None or best_sim < sim_threshold:
                new_role = get_role_label(next_role_id)
                next_role_id += 1

                role_feature_list[new_role].append(feat)
                role_centroids[new_role] = feat   # 首个样本直接作为中心
                role_images[new_role].add(img_name)
                continue

            # 3) 匹配到角色 → 添加样本并更新角色中心
            role_feature_list[matched_role].append(feat)

            # 更新角色 centroid（均值并归一化）
            new_centroid = np.mean(role_feature_list[matched_role], axis=0)
            role_centroids[matched_role] = normalize(new_centroid)

            role_images[matched_role].add(img_name)

    # 转换为普通 dict + list
    return {k: list(v) for k, v in role_images.items()}