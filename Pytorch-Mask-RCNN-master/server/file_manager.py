import pymysql
import os
import json
from datetime import datetime


class FileManager:
    def __init__(self, db_config=None, base_dir=None):
        if db_config is None:
            self.db_config = {
                'host': 'localhost',
                'user': 'root',
                'password': 'root',
                'db': 'maskrcnn',
                'charset': 'utf8mb4'
            }
        else:
            self.db_config = db_config
        
        if base_dir is None:
            import tempfile
            self.base_dir = tempfile.gettempdir()
        else:
            self.base_dir = base_dir
        
        self._init_tables()
    
    def _init_tables(self):
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INT PRIMARY KEY AUTO_INCREMENT,
                user_id INT NOT NULL,
                filename VARCHAR(255) NOT NULL,
                class_name VARCHAR(255),
                model_type VARCHAR(50),
                file_path VARCHAR(500) NOT NULL,
                file_size INT,
                created_at DATETIME NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INT PRIMARY KEY AUTO_INCREMENT,
                user_id INT NOT NULL,
                filename VARCHAR(255) NOT NULL,
                model_path VARCHAR(500) NOT NULL,
                box_conf REAL NOT NULL,
                mask_conf REAL NOT NULL,
                num_instances INT NOT NULL DEFAULT 0,
                instance_counts TEXT,
                created_at DATETIME NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INT PRIMARY KEY AUTO_INCREMENT,
                user_id INT NOT NULL,
                model_path VARCHAR(500) NOT NULL,
                bbox_map50 REAL,
                bbox_map50_95 REAL,
                bbox_map75 REAL,
                segm_map50 REAL,
                segm_map50_95 REAL,
                segm_map75 REAL,
                created_at DATETIME NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS camera_statistics (
                id INT PRIMARY KEY AUTO_INCREMENT,
                user_id INT NOT NULL,
                session_id VARCHAR(100) NOT NULL,
                model_path VARCHAR(500) NOT NULL,
                class_name VARCHAR(100) NOT NULL,
                total_count INT NOT NULL DEFAULT 0,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                duration_seconds INT DEFAULT 0,
                created_at DATETIME NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_user_session (user_id, session_id),
                INDEX idx_start_time (start_time)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_operations (
                id INT PRIMARY KEY AUTO_INCREMENT,
                username VARCHAR(255) NOT NULL,
                operation_type VARCHAR(50) NOT NULL,
                operation_time DATETIME NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'success',
                INDEX idx_username (username),
                INDEX idx_operation_time (operation_time)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_user_dir(self, user_id, subdir=None):
        user_dir = os.path.join(self.base_dir, f"user_{user_id}")
        if subdir:
            user_dir = os.path.join(user_dir, subdir)
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
    
    def add_model(self, user_id, filename, class_name, model_type, file_path):
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO models (user_id, filename, class_name, model_type, file_path, file_size, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (user_id, filename, class_name, model_type, file_path, file_size, created_at))
        conn.commit()
        model_id = cursor.lastrowid
        conn.close()
        return model_id
    
    def add_prediction(self, user_id, filename, model_path, box_conf, mask_conf, num_instances, instance_counts=None, result_path=None):
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        instance_counts_json = json.dumps(instance_counts) if instance_counts else None
        
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (user_id, filename, model_path, box_conf, mask_conf, num_instances, instance_counts, result_path, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (user_id, filename, model_path, box_conf, mask_conf, num_instances, instance_counts_json, result_path, created_at))
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        return prediction_id
    
    def add_evaluation(self, user_id, model_path, bbox_map50=None, bbox_map50_95=None, bbox_map75=None, segm_map50=None, segm_map50_95=None, segm_map75=None):
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO evaluations (user_id, model_path, bbox_map50, bbox_map50_95, bbox_map75, segm_map50, segm_map50_95, segm_map75, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (user_id, model_path, bbox_map50, bbox_map50_95, bbox_map75, segm_map50, segm_map50_95, segm_map75, created_at))
        conn.commit()
        eval_id = cursor.lastrowid
        conn.close()
        return eval_id
    
    def get_user_models(self, user_id):
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, filename, class_name, model_type, file_path, file_size, created_at
            FROM models WHERE user_id = %s ORDER BY created_at DESC
        ''', (user_id,))
        models = cursor.fetchall()
        conn.close()
        return models
    
    def get_user_predictions(self, user_id):
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, filename, model_path, box_conf, mask_conf, num_instances, instance_counts, created_at
            FROM predictions WHERE user_id = %s ORDER BY created_at DESC
        ''', (user_id,))
        predictions = cursor.fetchall()
        conn.close()
        return predictions
    
    def get_user_evaluations(self, user_id):
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, model_path, bbox_map50, bbox_map50_95, bbox_map75, segm_map50, segm_map50_95, segm_map75, created_at
            FROM evaluations WHERE user_id = %s ORDER BY created_at DESC
        ''', (user_id,))
        evaluations = cursor.fetchall()
        conn.close()
        return evaluations
    
    def save_camera_statistics(self, user_id, session_id, model_path, class_name, total_count, start_time, end_time=None, duration_seconds=0):
        """保存摄像头统计数据"""
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO camera_statistics 
            (user_id, session_id, model_path, class_name, total_count, start_time, end_time, duration_seconds, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (user_id, session_id, model_path, class_name, total_count, start_time, end_time, duration_seconds, created_at))
        conn.commit()
        stat_id = cursor.lastrowid
        conn.close()
        return stat_id
    
    def update_camera_statistics(self, session_id, class_name, total_count, end_time, duration_seconds):
        """更新摄像头统计数据（关闭摄像头时）"""
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE camera_statistics 
            SET total_count = %s, end_time = %s, duration_seconds = %s
            WHERE session_id = %s AND class_name = %s
        ''', (total_count, end_time, duration_seconds, session_id, class_name))
        conn.commit()
        conn.close()
    
    def get_user_camera_statistics(self, user_id, limit=100):
        """获取用户的摄像头统计数据"""
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, session_id, model_path, class_name, total_count, start_time, end_time, duration_seconds, created_at
            FROM camera_statistics 
            WHERE user_id = %s 
            ORDER BY start_time DESC 
            LIMIT %s
        ''', (user_id, limit))
        stats = cursor.fetchall()
        conn.close()
        return stats
    
    def get_session_statistics(self, session_id):
        """获取某个会话的统计数据"""
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, user_id, model_path, class_name, total_count, start_time, end_time, duration_seconds, created_at
            FROM camera_statistics 
            WHERE session_id = %s
            ORDER BY class_name
        ''', (session_id,))
        stats = cursor.fetchall()
        conn.close()
        return stats
    
    def log_user_operation(self, username, operation_type, status='success'):
        """记录用户操作"""
        operation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_operations (username, operation_type, operation_time, status)
            VALUES (%s, %s, %s, %s)
        ''', (username, operation_type, operation_time, status))
        conn.commit()
        conn.close()
    
    def get_user_operations(self, username=None, limit=100):
        """获取用户操作记录"""
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        
        if username:
            cursor.execute('''
                SELECT id, username, operation_type, operation_time, status
                FROM user_operations 
                WHERE username = %s
                ORDER BY operation_time DESC 
                LIMIT %s
            ''', (username, limit))
        else:
            cursor.execute('''
                SELECT id, username, operation_type, operation_time, status
                FROM user_operations 
                ORDER BY operation_time DESC 
                LIMIT %s
            ''', (limit,))
        
        operations = cursor.fetchall()
        conn.close()
        return operations
    
    def delete_model(self, model_id, user_id):
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('SELECT file_path FROM models WHERE id = %s AND user_id = %s', (model_id, user_id))
        result = cursor.fetchone()
        if result:
            file_path = result[0]
            if os.path.exists(file_path):
                os.remove(file_path)
            cursor.execute('DELETE FROM models WHERE id = %s', (model_id,))
            conn.commit()
        conn.close()
    
    def delete_prediction(self, prediction_id, user_id):
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM predictions WHERE id = %s AND user_id = %s', (prediction_id, user_id))
        conn.commit()
        conn.close()
    
    def delete_evaluation(self, eval_id, user_id):
        conn = pymysql.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM evaluations WHERE id = %s AND user_id = %s', (eval_id, user_id))
        conn.commit()
        conn.close()
