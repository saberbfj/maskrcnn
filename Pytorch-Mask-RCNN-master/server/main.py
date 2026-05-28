import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import uvicorn
from server.user_manager import UserManager
from server.file_manager import FileManager

app = FastAPI(title="Mask R-CNN Instance Segmentation API")

file_manager = FileManager()
user_manager = UserManager(file_manager=file_manager)

from server.train_api import router as train_router
from server.predict_api import router as predict_router
from server.evaluate_api import router as evaluate_router
from server.camera_api import router as camera_router

app.include_router(train_router)
app.include_router(predict_router)
app.include_router(evaluate_router)
app.include_router(camera_router)

def verify_user(username, password):
    """验证用户"""
    success, message = user_manager.login_user(username, password)
    if not success:
        raise ValueError(message)
    user_id = user_manager.get_user_id(username)
    is_admin = user_manager.is_admin(username)
    return {"user_id": user_id, "username": username, "is_admin": is_admin}

@app.post("/api/auth/login")
def login(username: str = Form(...), password: str = Form(...)):
    """用户登录"""
    try:
        user_info = verify_user(username, password)
        return {
            "success": True,
            "message": "登录成功",
            "user_id": user_info["user_id"],
            "username": user_info["username"],
            "is_admin": user_info["is_admin"]
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}

@app.post("/api/auth/register")
def register(username: str = Form(...), password: str = Form(...)):
    """用户注册"""
    success, message = user_manager.register_user(username, password)
    if success:
        user_id = user_manager.get_user_id(username)
        return {"success": True, "message": message, "user_id": user_id}
    return {"success": False, "error": message}

@app.get("/api/users")
def get_users(username: str = None, password: str = None):
    """获取用户列表（仅管理员）"""
    try:
        user_info = verify_user(username, password)
        if not user_info["is_admin"]:
            return JSONResponse(status_code=403, content={"error": "只有管理员可以查看用户列表"})

        users = user_manager.get_all_users()
        user_list = [{
            "id": u["id"],
            "username": u["username"],
            "created_at": u["created_at"],
            "last_login": u["last_login"],
            "is_admin": u["is_admin"]
        } for u in users]

        return {"success": True, "users": user_list}
    except ValueError as e:
        return JSONResponse(status_code=401, content={"error": str(e)})

@app.delete("/api/users/{user_id}")
def delete_user(user_id: int, username: str = None, password: str = None):
    """删除用户（仅管理员）"""
    try:
        user_info = verify_user(username, password)
        if not user_info["is_admin"]:
            return JSONResponse(status_code=403, content={"error": "只有管理员可以删除用户"})

        success, message = user_manager.delete_user(user_id)
        if success:
            return {"success": True, "message": message}
        return JSONResponse(status_code=400, content={"error": message})
    except ValueError as e:
        return JSONResponse(status_code=401, content={"error": str(e)})

@app.post("/api/users/{user_id}/password")
def change_user_password(
    user_id: int,
    username: str = Form(...),
    password: str = Form(...),
    new_password: str = Form(...)
):
    """修改用户密码（仅管理员）"""
    try:
        user_info = verify_user(username, password)
        if not user_info["is_admin"]:
            return JSONResponse(status_code=403, content={"error": "只有管理员可以修改用户密码"})

        success, message = user_manager.change_password(user_id, new_password)
        if success:
            return {"success": True, "message": message}
        return JSONResponse(status_code=400, content={"error": message})
    except ValueError as e:
        return JSONResponse(status_code=401, content={"error": str(e)})

@app.get("/api/models")
def get_models(username: str = None, password: str = None):
    """获取用户模型列表"""
    try:
        if username and password:
            user_info = verify_user(username, password)
            user_id = user_info["user_id"]
            models = file_manager.get_user_models(user_id)
            return {"success": True, "models": models}
        return {"success": False, "error": "缺少认证信息"}
    except ValueError as e:
        return JSONResponse(status_code=401, content={"error": str(e)})

@app.get("/api/predictions")
def get_predictions(username: str = None, password: str = None):
    """获取用户预测结果列表"""
    try:
        if username and password:
            user_info = verify_user(username, password)
            user_id = user_info["user_id"]
            predictions = file_manager.get_user_predictions(user_id)
            return {"success": True, "predictions": predictions}
        return {"success": False, "error": "缺少认证信息"}
    except ValueError as e:
        return JSONResponse(status_code=401, content={"error": str(e)})

@app.get("/api/files/{file_path:path}")
def get_file(file_path: str, username: str = None, password: str = None):
    """获取文件"""
    try:
        if username and password:
            user_info = verify_user(username, password)
            full_path = file_manager.get_user_dir(user_info["user_id"])
            return full_path
        return None
    except ValueError as e:
        return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
