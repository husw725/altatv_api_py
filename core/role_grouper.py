import os
from collections import defaultdict
from utils.face_utils import extract_feature, cosine_sim, get_role_label

SIM_THRESHOLD = 0.6  # 人脸特征相似度阈值


def group_roles(
    input_dir: str,
    det_threshold: float = 0.75,
    sim_threshold: float = SIM_THRESHOLD
) -> dict:
    """
    按人脸特征聚类分组（不保存图片）
    Args:
        input_dir: 本地图片目录
        det_threshold: 人脸检测置信度阈值
        sim_threshold: 特征相似度阈值
    Returns:
        dict: {角色名: [图片文件名列表]}
    """
    role_features = {}          # 每个角色的参考特征
    role_images = defaultdict(list)
    next_role_id = 0

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # 提取图片中的人脸特征
        features = extract_feature(img_path, det_threshold=det_threshold)
        if not features:
            role_images["other"].append(img_name)
            continue

        for feat in features:
            matched_role = None
            best_sim = 0.0

            # 与已知角色特征比较
            for role, ref_feat in role_features.items():
                sim = cosine_sim(feat, ref_feat)
                if sim > best_sim:
                    best_sim = sim
                    matched_role = role

            # 若找到相似角色则归入，否则新建角色
            if matched_role and best_sim >= sim_threshold:
                role_images[matched_role].append(img_name)
            else:
                new_role = get_role_label(next_role_id)
                role_features[new_role] = feat
                role_images[new_role].append(img_name)
                next_role_id += 1

    # 返回角色分组结果
    return dict(role_images)