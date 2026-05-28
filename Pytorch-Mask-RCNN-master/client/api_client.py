import requests
import json
import os
from typing import List, Dict, Optional, Any


class APIClient:
    def __init__(self, base_url: str = None):
        # 从配置文件读取服务器地址
        if base_url is None:
            base_url = self.load_server_url()
        self.base_url = base_url
        self.session = requests.Session()
        self.token = None
        self.user_info = None
    
    def load_server_url(self):
        """从配置文件加载服务器地址"""
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("server_url", "http://localhost:8000")
        except:
            return "http://localhost:8000"
    
    def login(self, username: str, password: str) -> Dict:
        """登录"""
        url = f"{self.base_url}/api/auth/login"
        data = {
            "username": username,
            "password": password
        }
        response = self.session.post(url, data=data)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                self.user_info = {
                    "user_id": result.get("user_id"),
                    "username": result.get("username"),
                    "is_admin": result.get("is_admin"),
                    "password": password  # 保存密码用于后续API调用
                }
                return result
        return {"success": False, "error": response.json().get("error", "登录失败")}
    
    def register(self, username: str, password: str) -> Dict:
        """注册"""
        url = f"{self.base_url}/api/auth/register"
        data = {
            "username": username,
            "password": password
        }
        response = self.session.post(url, data=data)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.json().get("error", "注册失败")}
    
    def get_users(self) -> Dict:
        """获取用户列表（仅管理员）"""
        if not self.user_info or not self.user_info.get("is_admin"):
            return {"success": False, "error": "只有管理员可以查看用户列表"}
        
        url = f"{self.base_url}/api/users"
        params = {
            "username": self.user_info.get("username"),
            "password": self.user_info.get("password")
        }
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.json().get("error", "获取用户列表失败")}
    
    def delete_user(self, user_id: int) -> Dict:
        """删除用户（仅管理员）"""
        if not self.user_info or not self.user_info.get("is_admin"):
            return {"success": False, "error": "只有管理员可以删除用户"}
        
        url = f"{self.base_url}/api/users/{user_id}"
        params = {
            "username": self.user_info.get("username"),
            "password": self.user_info.get("password")
        }
        response = self.session.delete(url, params=params)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.json().get("error", "删除用户失败")}
    
    def change_user_password(self, user_id: int, new_password: str) -> Dict:
        """修改用户密码（仅管理员）"""
        if not self.user_info or not self.user_info.get("is_admin"):
            return {"success": False, "error": "只有管理员可以修改用户密码"}
        
        url = f"{self.base_url}/api/users/{user_id}/password"
        data = {
            "username": self.user_info.get("username"),
            "password": self.user_info.get("password"),
            "new_password": new_password
        }
        response = self.session.post(url, data=data)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.json().get("error", "修改密码失败")}
    
    def start_training(self, dataset_dir: str, save_dir: str = None) -> Dict:
        """开始训练"""
        if not self.user_info:
            return {"success": False, "error": "请先登录"}
        
        url = f"{self.base_url}/api/train/start"
        data = {
            "username": self.user_info.get("username"),
            "password": self.user_info.get("password"),
            "dataset_dir": dataset_dir,
            "save_dir": save_dir
        }
        response = self.session.post(url, data=data)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.json().get("error", "开始训练失败")}
    
    def get_train_status(self, task_id: str) -> Dict:
        """获取训练状态"""
        if not self.user_info:
            return {"success": False, "error": "请先登录"}
        
        url = f"{self.base_url}/api/train/status/{task_id}"
        params = {
            "username": self.user_info.get("username"),
            "password": self.user_info.get("password")
        }
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.json().get("error", "获取训练状态失败")}
    
    def stop_training(self, task_id: str) -> Dict:
        """停止训练"""
        if not self.user_info:
            return {"success": False, "error": "请先登录"}
        
        url = f"{self.base_url}/api/train/stop"
        data = {
            "task_id": task_id,
            "username": self.user_info.get("username"),
            "password": self.user_info.get("password")
        }
        response = self.session.post(url, data=data)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.json().get("error", "停止训练失败")}
    
    def predict(self, image_path: str, model_paths: List[str], save_dir: str = None, box_conf: float = 0.3, mask_conf: float = 0.3) -> Any:
        """预测图片"""
        if not self.user_info:
            return {"success": False, "error": "请先登录"}
        
        url = f"{self.base_url}/api/predict"
        try:
            with open(image_path, "rb") as f:
                files = {
                    "image": (os.path.basename(image_path), f)
                }
                data = {
                    "model_paths": json.dumps(model_paths),
                    "save_dir": save_dir,
                    "box_conf": str(box_conf),
                    "mask_conf": str(mask_conf),
                    "username": self.user_info.get("username"),
                    "password": self.user_info.get("password")
                }
                response = self.session.post(url, files=files, data=data)
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if result.get("success"):
                            # 返回包含image和model_results的字典
                            return {
                                "success": True,
                                "image": result.get("image"),
                                "model_results": result.get("model_results", [])
                            }
                        else:
                            return {"success": False, "error": result.get("error", "预测失败")}
                    except Exception as e:
                        return {"success": False, "error": f"解析响应失败: {str(e)}"}
                else:
                    try:
                        error_data = response.json()
                        return {"success": False, "error": error_data.get("error", f"请求失败，状态码: {response.status_code}")}
                    except Exception:
                        return {"success": False, "error": f"请求失败，状态码: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"发送请求失败: {str(e)}"}
    
    def predict_base64(self, image_base64: str, model_paths: List[str], box_conf: float = 0.3, mask_conf: float = 0.3) -> Any:
        """预测base64编码的图像（摄像头用）"""
        if not self.user_info:
            return {"success": False, "error": "请先登录"}
        
        url = f"{self.base_url}/api/predict/camera"
        try:
            data = {
                "image_base64": image_base64,
                "model_paths": json.dumps(model_paths),
                "box_conf": str(box_conf),
                "mask_conf": str(mask_conf),
                "username": self.user_info.get("username"),
                "password": self.user_info.get("password")
            }
            response = self.session.post(url, data=data)
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get("success"):
                        # 返回包含image和model_results的字典
                        return {
                            "success": True,
                            "image": result.get("image"),
                            "model_results": result.get("model_results", [])
                        }
                    else:
                        return {"success": False, "error": result.get("error", "预测失败")}
                except Exception as e:
                    return {"success": False, "error": f"解析响应失败: {str(e)}"}
            else:
                try:
                    error_data = response.json()
                    return {"success": False, "error": error_data.get("error", f"请求失败，状态码: {response.status_code}")}
                except Exception:
                    return {"success": False, "error": f"请求失败，状态码: {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": f"发送请求失败: {str(e)}"}
    
    def evaluate(self, test_dir: str, weights_path: str, save_dir: str = None) -> Dict:
        """评估模型"""
        if not self.user_info:
            return {"success": False, "error": "请先登录"}
        
        url = f"{self.base_url}/api/evaluate"
        data = {
            "test_dir": test_dir,
            "weights_path": weights_path,
            "save_dir": save_dir,
            "username": self.user_info.get("username"),
            "password": self.user_info.get("password")
        }
        response = self.session.post(url, data=data)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.json().get("error", "评估失败")}
    
    def get_models(self) -> Dict:
        """获取用户模型列表"""
        if not self.user_info:
            return {"success": False, "error": "请先登录"}
        
        url = f"{self.base_url}/api/models"
        params = {
            "username": self.user_info.get("username"),
            "password": self.user_info.get("password")
        }
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.json().get("error", "获取模型列表失败")}
    
    def get_predictions(self) -> Dict:
        """获取用户预测结果列表"""
        if not self.user_info:
            return {"success": False, "error": "请先登录"}
        
        url = f"{self.base_url}/api/predictions"
        params = {
            "username": self.user_info.get("username"),
            "password": self.user_info.get("password")
        }
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.json().get("error", "获取预测结果列表失败")}
    
    def get_evaluations(self) -> Dict:
        """获取用户评估结果列表"""
        if not self.user_info:
            return {"success": False, "error": "请先登录"}
        
        url = f"{self.base_url}/api/evaluations"
        params = {
            "username": self.user_info.get("username"),
            "password": self.user_info.get("password")
        }
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.json().get("error", "获取评估结果列表失败")}
    
    def delete_evaluation(self, eval_id: int) -> Dict:
        """删除评估记录"""
        if not self.user_info:
            return {"success": False, "error": "请先登录"}
        
        url = f"{self.base_url}/api/evaluations/{eval_id}"
        params = {
            "username": self.user_info.get("username"),
            "password": self.user_info.get("password")
        }
        response = self.session.delete(url, params=params)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.json().get("error", "删除评估记录失败")}
    
    def get_file(self, file_path: str) -> Any:
        """获取文件"""
        if not self.user_info:
            return {"success": False, "error": "请先登录"}
        
        url = f"{self.base_url}/api/files/{file_path}"
        params = {
            "username": self.user_info.get("username"),
            "password": self.user_info.get("password")
        }
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.content
        return {"success": False, "error": response.json().get("error", "获取文件失败")}
    
    def logout(self):
        """登出"""
        self.session.close()
        self.token = None
        self.user_info = None