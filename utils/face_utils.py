import insightface
import numpy as np
import os
import cv2

model_name = 'buffalo_l' # 'antelopev2'#buffalo_l
# 初始化模型
model = insightface.app.FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

NORM_THRESHOLD = 0.5 # 人脸特征向量范数过滤阈值
DET_THRESHOLD = 0.75  # 人脸检测置信度过滤阈值


# ------------------ 工具函数 ------------------
def extract_feature(image_path,
                    norm_threshold=NORM_THRESHOLD,
                    det_threshold=DET_THRESHOLD):
    """提取人脸特征，过滤掉低质量人脸"""
    img = cv2.imread(image_path)
    if img is None:
        return []

    faces = model.get(img)
    if not faces:
        return []

    features = []
    for f in faces:
        if f.det_score is not None and f.det_score < det_threshold:
            continue  # 置信度太低
        emb = f.embedding
        if emb is None:
            continue
        norm = np.linalg.norm(emb)
        if norm < norm_threshold:  # 特征向量太小/太弱
            continue
        print(f.det_score, norm,image_path)
        features.append(emb)
    return features


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def get_role_label(idx: int) -> str:
    label = ""
    while True:
        idx, rem = divmod(idx, 26)
        label = chr(ord("A") + rem) + label
        if idx == 0:
            break
        idx -= 1
    return label