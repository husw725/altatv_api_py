from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 导入路由
# from api.scene_api import router as scene_router
from api.role_api import router as role_router

# ------------------ 初始化应用 ------------------
app = FastAPI(
    title="Video Scene + Face Role API",
    description="提供视频智能分镜和人物分组接口",
    version="1.0.0"
)

# ------------------ 可选：跨域配置 ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允许所有域名访问，可按需修改
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ 注册路由 ------------------
# app.include_router(scene_router, prefix="/api/scene", tags=["SceneDetect"])
app.include_router(role_router, prefix="/api/role", tags=["RoleGrouping"])

# ------------------ 根路由 ------------------
@app.get("/")
def root():
    return {"message": "Video Scene + Face Role API is running."}

# ------------------ 启动 ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)