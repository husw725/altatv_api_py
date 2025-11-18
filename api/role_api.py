from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os, tempfile, requests, shutil
from core.role_grouper import group_roles

router = APIRouter()

class GroupRequest(BaseModel):
    image_urls: List[str]     # 图片 URL 列表
    # output_dir: str           # 输出目录
    det_threshold: float = 0.55

def download_images(urls: List[str], tmp_dir: str) -> dict:
    """
    下载图片，返回 {本地文件名: 原始URL} 映射
    """
    os.makedirs(tmp_dir, exist_ok=True)
    local2url = {}
    for i, url in enumerate(urls):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                ext = ".jpg" if "jpeg" in r.headers.get("content-type", "") else ".png"
                filename = f"img_{i}{ext}"
                filepath = os.path.join(tmp_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(r.content)
                local2url[filename] = url
        except Exception as e:
            print(f"⚠️ 下载失败 {url}: {e}")
    return local2url

@router.post("/group_roles")
def group_roles_api(req: GroupRequest):
    if not req.image_urls:
        raise HTTPException(status_code=400, detail="图片 URL 列表为空")

    tmp_dir = tempfile.mkdtemp(prefix="roles_input_")

    try:
        # Step 1: 下载图片
        local2url = download_images(req.image_urls, tmp_dir)
        input_dir = tmp_dir

        # Step 2: 分组
        roles = group_roles(input_dir, sim_threshold=req.det_threshold) # det_threshold 当sim_threshold来用,不影响api调用

        # Step 3: 用原始 URL 生成返回结果
        result_roles = {}
        for role, images in roles.items():
            result_roles[role] = [local2url.get(os.path.basename(img), "") for img in images]

        result = {
            "status": "success",
            "role_count": len(result_roles),
            "roles": result_roles
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


@router.get("/test")
def group_roles_api():
    return {"message": "test"}