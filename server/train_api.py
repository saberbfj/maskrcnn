import torch
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
import os
import threading
import time
from datetime import datetime
from detection.train_script import train_model

router = APIRouter()

# 存储训练任务状态
training_tasks = {}

class TrainingTask:
    def __init__(self, task_id, dataset_dir, save_dir, user_id):
        self.task_id = task_id
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.user_id = user_id
        self.status = "running"
        self.epoch = 0
        self.progress = 0
        self.logs = []
        self.loss_history = []
        self.stop_flag = False
        self.start_time = time.time()
        self.thread = None

    def stop(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=5)

    def should_stop(self):
        return self.stop_flag

    def update(self, progress, message, loss_data=None):
        self.progress = progress
        self.logs.append(message)
        # 打印到终端，显示详细训练进度
        print(f"[训练] {message}")
        if loss_data:
            self.loss_history.append(loss_data)

@router.post("/api/train/start")
def start_training(
    username: str = Form(...),
    password: str = Form(...),
    dataset_dir: str = Form(...),
    save_dir: str = Form(None)
):
    """开始模型训练"""
    from server.main import user_manager, file_manager
    
    success, message = user_manager.login_user(username, password)
    if not success:
        return JSONResponse(status_code=401, content={"error": message})
    
    user_id = user_manager.get_user_id(username)
    
    if not os.path.exists(dataset_dir):
        return JSONResponse(status_code=400, content={"error": "数据集目录不存在"})
    
    train_images_dir = os.path.join(dataset_dir, "train", "PNGImages")
    train_masks_dir = os.path.join(dataset_dir, "train", "PedMasks")
    val_images_dir = os.path.join(dataset_dir, "val", "PNGImages")
    val_masks_dir = os.path.join(dataset_dir, "val", "PedMasks")
    
    if not all(os.path.exists(d) for d in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]):
        return JSONResponse(status_code=400, content={"error": "数据集目录结构不完整"})
    
    if not save_dir:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "user_files", username, "model")
    os.makedirs(save_dir, exist_ok=True)
    
    task_id = f"train_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建训练任务
    task = TrainingTask(task_id, dataset_dir, save_dir, user_id)
    training_tasks[task_id] = task
    
    # 启动训练线程
    def training_thread_func():
        try:
            task.update(0, f"训练任务开始: {task_id}")
            task.update(0, f"使用设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            
            # 运行训练
            result = train_model(
                dataset_dir=dataset_dir,
                save_dir=save_dir,
                user_id=user_id,
                file_manager=file_manager,
                update_callback=task.update,
                stop_check=task.should_stop
            )
            
            if result:
                task.update(100, "训练完成!")
                task.status = "completed"
            else:
                task.update(100, "训练失败")
                task.status = "error"
        except Exception as e:
            task.update(100, f"训练过程中出现错误: {str(e)}")
            task.status = "error"
    
    task.thread = threading.Thread(target=training_thread_func)
    task.thread.daemon = True
    task.thread.start()
    
    return {
        "success": True,
        "message": "训练任务已启动",
        "task_id": task_id,
        "save_dir": save_dir
    }

@router.get("/api/train/status/{task_id}")
def get_training_status(task_id: str, username: str = None, password: str = None):
    """获取训练状态"""
    if username and password:
        from server.main import user_manager
        success, message = user_manager.login_user(username, password)
        if not success:
            return JSONResponse(status_code=401, content={"error": message})
    
    task = training_tasks.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "训练任务不存在"})
    
    return {
        "success": True,
        "task": {
            "task_id": task.task_id,
            "status": task.status,
            "epoch": task.epoch,
            "progress": task.progress,
            "logs": task.logs,
            "loss_history": task.loss_history
        }
    }

@router.post("/api/train/stop")
def stop_training(
    task_id: str = Form(...),
    username: str = Form(...),
    password: str = Form(...)
):
    """停止训练"""
    from server.main import user_manager
    
    success, message = user_manager.login_user(username, password)
    if not success:
        return JSONResponse(status_code=401, content={"error": message})
    
    task = training_tasks.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "训练任务不存在"})
    
    task.stop()
    task.status = "stopped"
    task.update(100, "训练已停止")
    
    return {
        "success": True,
        "message": "训练已停止"
    }
