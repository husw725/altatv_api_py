import os
import numpy as np
import cv2
from collections import defaultdict
from utils.face_utils import extract_feature, cosine_sim, get_role_label


# -------------------------
# 默认参数
# -------------------------
SIM_THRESHOLD = 0.55
DET_THRESHOLD = 0.65


# -------------------------
# 工具函数
# -------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalize feature"""
    return v / (np.linalg.norm(v) + 1e-6)


def compute_clarity(img_path: str) -> float:
    """
    返回 0~1，值越大越清晰
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.3

    lap = cv2.Laplacian(img, cv2.CV_64F)
    score = lap.var()

    score = min(score / 1500.0, 1.0)
    return max(score, 0.05)


# -------------------------
# 第 1 阶段：初步聚类
# -------------------------
def _first_pass(
    file_list,
    input_dir,
    det_threshold,
    sim_threshold,
    feature_cache
):
    role_centroids = {}
    role_feature_list = defaultdict(list)
    role_images = defaultdict(set)
    next_role_id = 0

    for img_name in file_list:
        img_path = os.path.join(input_dir, img_name)

        # --- 特征缓存防重复计算 ---
        if img_name not in feature_cache:
            features = extract_feature(img_path, det_threshold=det_threshold)
            feature_cache[img_name] = features
        else:
            features = feature_cache[img_name]

        if not features:
            role_images["other"].add(img_name)
            continue

        clarity = compute_clarity(img_path)

        for feat in features:
            feat = normalize(np.array(feat))

            matched_role = None
            best_sim = -1

            # --- 与所有 centroid 比较 ---
            for role, centroid in role_centroids.items():
                sim = cosine_sim(feat, centroid)
                if sim > best_sim:
                    best_sim = sim
                    matched_role = role

            # --- 自适应阈值（清晰度越高阈值越严格） ---
            adaptive_thr = sim_threshold * (0.8 + clarity * 0.2)

            # --- 新角色 ---
            if best_sim < adaptive_thr:
                new_role = get_role_label(next_role_id)
                next_role_id += 1

                role_feature_list[new_role].append((feat, clarity))
                role_centroids[new_role] = feat
                role_images[new_role].add(img_name)
                continue

            # --- 更新已有角色 centroid（加权） ---
            role_feature_list[matched_role].append((feat, clarity))

            feats = np.array([f for f, w in role_feature_list[matched_role]])
            weights = np.array([w**2 for f, w in role_feature_list[matched_role]])

            new_centroid = normalize(np.average(feats, axis=0, weights=weights))
            role_centroids[matched_role] = new_centroid

            role_images[matched_role].add(img_name)

    return role_centroids, role_feature_list, role_images


# -------------------------
# 第 2 阶段：重新聚类（精炼）
# -------------------------
def _second_pass(
    file_list,
    input_dir,
    centroids,
    feature_cache,
    sim_threshold
):
    final_groups = defaultdict(set)

    for img_name in file_list:
        img_path = os.path.join(input_dir, img_name)

        features = feature_cache.get(img_name)
        if not features:
            final_groups["other"].add(img_name)
            continue

        clarity = compute_clarity(img_path)

        for feat in features:
            feat = normalize(np.array(feat))

            matched_role = None
            best_sim = -1

            for role, centroid in centroids.items():
                sim = cosine_sim(feat, centroid)
                if sim > best_sim:
                    best_sim = sim
                    matched_role = role

            adaptive_thr = sim_threshold * (0.8 + clarity * 0.2)

            if best_sim >= adaptive_thr:
                final_groups[matched_role].add(img_name)
            else:
                final_groups["other"].add(img_name)

    return final_groups


# -------------------------
# 主函数：两阶段聚类（直接可运行）
# -------------------------
def group_roles(
    input_dir: str,
    det_threshold: float = DET_THRESHOLD,
    sim_threshold: float = SIM_THRESHOLD
) -> dict:
    """
    按人脸特征聚类分组
    - 2 阶段聚类（更准确）
    - 清晰度加权
    - 自适应阈值
    - 特征缓存
    """
    file_list = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    feature_cache = {}

    # --- 第一阶段：建立 rough centroid ---
    centroids, role_feature_list, role_images = _first_pass(
        file_list,
        input_dir,
        det_threshold,
        sim_threshold,
        feature_cache
    )

    # --- 第二阶段：根据最终 centroid 重新分组 ---
    final_groups = _second_pass(
        file_list,
        input_dir,
        centroids,
        feature_cache,
        sim_threshold
    )

    return {k: list(v) for k, v in final_groups.items()}