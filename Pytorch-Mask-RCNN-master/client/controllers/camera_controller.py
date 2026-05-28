from PyQt5.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import io
import time

class CameraThread(QThread):
    preview_signal = pyqtSignal(object)
    result_signal = pyqtSignal(object)
    log_signal = pyqtSignal(str)
    update_signal = pyqtSignal(str)

    def __init__(self, api_client, model_paths, box_conf, mask_conf, camera_index=0, interval=1.0):
        super().__init__()
        self.api_client = api_client
        self.model_paths = model_paths
        self.box_conf = box_conf
        self.mask_conf = mask_conf
        self.camera_index = camera_index
        self.running = True
        self.interval = interval
        self.last_log_time = 0
        self.interval_instance_counts = {}
        self.interval_predictions = 0

    def run(self):
        try:
            import cv2
            import base64
            from PIL import Image
            import os
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                self.update_signal.emit(f"无法打开摄像头 {self.camera_index}")
                return
            self.update_signal.emit(f"摄像头 {self.camera_index} 已启动")
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
                self.preview_signal.emit(frame_rgb)
                try:
                    pil_image = Image.fromarray(frame_rgb)
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='JPEG')
                    image_bytes = buffer.getvalue()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    result = self.api_client.predict_base64(
                        image_base64, self.model_paths, self.box_conf, self.mask_conf
                    )
                    if result.get("success"):
                        result_image_base64 = result.get("image")
                        model_results = result.get("model_results", [])
                        self.interval_predictions += 1
                        for model_result in model_results:
                            model_name = model_result.get("model", "未知模型")
                            num_instances = model_result.get("boxes", 0)
                            if model_name in self.interval_instance_counts:
                                self.interval_instance_counts[model_name] += num_instances
                            else:
                                self.interval_instance_counts[model_name] = num_instances
                        if result_image_base64:
                            result_image_bytes = base64.b64decode(result_image_base64)
                            result_buffer = io.BytesIO(result_image_bytes)
                            result_pil_image = Image.open(result_buffer)
                            result_frame = np.array(result_pil_image).copy()
                            self.result_signal.emit(result_frame)
                        current_time = time.time()
                        if current_time - self.last_log_time >= self.interval:
                            self.last_log_time = current_time
                            log_message = ""
                            if self.interval_instance_counts:
                                total_instances = sum(self.interval_instance_counts.values())
                                log_message = f"过去{self.interval}秒预测结果（共{self.interval_predictions}次预测）:\n"
                                for model_name, count in self.interval_instance_counts.items():
                                    log_message += f"- {model_name}: {count} 个实例\n"
                                log_message += f"总计: {total_instances} 个实例"
                            else:
                                log_message = f"过去{self.interval}秒没有检测到实例"
                            self.log_signal.emit(log_message)
                            self.interval_instance_counts = {}
                            self.interval_predictions = 0
                    else:
                        self.log_signal.emit(f"预测失败: {result.get('error', '未知错误')}")
                except Exception as e:
                    self.log_signal.emit(f"预测错误: {str(e)}")
                self.msleep(50)
            cap.release()
            self.update_signal.emit("摄像头已关闭")
        except Exception as e:
            self.update_signal.emit(f"摄像头错误: {str(e)}")

    def stop(self):
        self.running = False


class CameraController:
    def __init__(self, ui_widget, api_client):
        self.ui = ui_widget
        self.api_client = api_client
        self.camera_thread = None
        self._setup_connections()

    def _setup_connections(self):
        self.ui.add_model_btn.clicked.connect(self.add_model)
        self.ui.remove_model_btn.clicked.connect(self.remove_model)
        self.ui.start_camera_btn.clicked.connect(self.start_camera)
        self.ui.stop_camera_btn.clicked.connect(self.stop_camera)

    def add_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self.ui, "选择模型权重文件", "", "模型文件 (*.pth)")
        if file_path:
            for i in range(self.ui.camera_weights_list.count()):
                if self.ui.camera_weights_list.item(i).text() == file_path:
                    QMessageBox.information(self.ui, "提示", "该模型已经添加")
                    return
            self.ui.camera_weights_list.addItem(file_path)

    def remove_model(self):
        selected_items = self.ui.camera_weights_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self.ui, "提示", "请先选择要移除的模型")
            return
        for item in selected_items:
            self.ui.camera_weights_list.takeItem(self.ui.camera_weights_list.row(item))

    def get_interval_seconds(self):
        interval_text = self.ui.camera_interval_combo.currentText()
        if "0.5秒" in interval_text:
            return 0.5
        elif "1分钟" in interval_text:
            return 60
        elif "秒" in interval_text:
            return float(interval_text.replace("秒", ""))
        return 1.0

    def start_camera(self):
        if not self.ui.camera_weights_list.count():
            QMessageBox.warning(self.ui, "提示", "请添加至少一个模型权重")
            return
        model_paths = []
        for i in range(self.ui.camera_weights_list.count()):
            item = self.ui.camera_weights_list.item(i)
            model_paths.append(item.text())
        try:
            box_conf = float(self.ui.camera_box_conf_input.text())
            mask_conf = float(self.ui.camera_mask_conf_input.text())
        except ValueError:
            QMessageBox.warning(self.ui, "提示", "请输入有效的置信度值")
            return
        camera_index = self.ui.camera_selector.currentIndex()
        interval = self.get_interval_seconds()
        self.ui.start_camera_btn.setEnabled(False)
        self.ui.stop_camera_btn.setEnabled(True)
        self.ui.camera_log.clear()
        self.camera_thread = CameraThread(
            self.api_client, model_paths, box_conf, mask_conf, camera_index, interval
        )
        self.camera_thread.preview_signal.connect(self.display_preview)
        self.camera_thread.result_signal.connect(self.display_result)
        self.camera_thread.log_signal.connect(self.update_log)
        self.camera_thread.update_signal.connect(self.update_status)
        self.camera_thread.start()

    def stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
        self.ui.start_camera_btn.setEnabled(True)
        self.ui.stop_camera_btn.setEnabled(False)

    def display_preview(self, frame):
        try:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.ui.camera_original_label.width(),
                self.ui.camera_original_label.height(),
                aspectRatioMode=1
            )
            self.ui.camera_original_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"显示预览失败: {e}")

    def display_result(self, frame):
        try:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.ui.camera_result_label.width(),
                self.ui.camera_result_label.height(),
                aspectRatioMode=1
            )
            self.ui.camera_result_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"显示结果失败: {e}")

    def update_log(self, message):
        self.ui.camera_log.append(message)

    def update_status(self, message):
        self.ui.camera_log.append(message)

    def cleanup(self):
        self.stop_camera()
