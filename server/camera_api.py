from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/api/camera/save_statistics")
def save_camera_statistics(
    username: str = Form(...),
    password: str = Form(...),
    session_id: str = Form(...),
    model_path: str = Form(...),
    class_name: str = Form(...),
    total_count: int = Form(...),
    start_time: str = Form(...)
):
    """保存摄像头统计数据"""
    from server.main import user_manager, file_manager
    
    success, message = user_manager.login_user(username, password)
    if not success:
        return JSONResponse(status_code=401, content={"error": message})
    
    user_id = user_manager.get_user_id(username)
    
    file_manager.save_camera_statistics(
        user_id=user_id,
        session_id=session_id,
        model_path=model_path,
        class_name=class_name,
        total_count=total_count,
        start_time=start_time
    )
    
    return {"success": True, "message": "统计数据已保存"}

@router.post("/api/camera/update_statistics")
def update_camera_statistics(
    username: str = Form(...),
    password: str = Form(...),
    session_id: str = Form(...),
    class_name: str = Form(...),
    total_count: int = Form(...),
    end_time: str = Form(...),
    duration_seconds: int = Form(...)
):
    """更新摄像头统计数据"""
    from server.main import user_manager, file_manager
    
    success, message = user_manager.login_user(username, password)
    if not success:
        return JSONResponse(status_code=401, content={"error": message})
    
    file_manager.update_camera_statistics(
        session_id=session_id,
        class_name=class_name,
        total_count=total_count,
        end_time=end_time,
        duration_seconds=duration_seconds
    )
    
    return {"success": True, "message": "统计数据已更新"}

@router.get("/api/camera/statistics")
def get_camera_statistics(username: str = None, password: str = None, limit: int = 100):
    """获取摄像头统计数据"""
    if username and password:
        from server.main import user_manager, file_manager
        success, message = user_manager.login_user(username, password)
        if not success:
            return JSONResponse(status_code=401, content={"error": message})
        
        user_id = user_manager.get_user_id(username)
        stats = file_manager.get_user_camera_statistics(user_id, limit)
        return {"success": True, "statistics": stats}
    return {"success": False, "error": "缺少认证信息"}

@router.get("/api/camera/session/{session_id}")
def get_session_statistics(session_id: str, username: str = None, password: str = None):
    """获取某个会话的统计数据"""
    if username and password:
        from server.main import user_manager, file_manager
        success, message = user_manager.login_user(username, password)
        if not success:
            return JSONResponse(status_code=401, content={"error": message})
        
        stats = file_manager.get_session_statistics(session_id)
        return {"success": True, "statistics": stats}
    return {"success": False, "error": "缺少认证信息"}
