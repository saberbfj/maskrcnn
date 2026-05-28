from PyQt5.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from PIL import Image
import io

class PredictThread(QThread):
    update_signal = pyqtSignal(str, object)
    finished_signal = pyqtSignal(object, object)

    def __init__(self, api_client, image_path, model_paths, save_dir, box_conf, mask_conf):
        super().__init__()
        self.api_client = api_client
        self.image_path = image_path
        self.model_paths = model_paths
        self.save_dir = save_dir
        self.box_conf = box_conf
        self.mask_conf = mask_conf

    def run(self):
        try:
            self.update_signal.emit("开始预测...", None)
            result = self.api_client.predict(
                self.image_path,
                self.model_paths,
                self.save_dir,
                self.box_conf,
                self.mask_conf
            )
            if isinstance(result, dict) and result.get("success"):
                model_results = result.get("model_results", [])
                for i, mr in enumerate(model_results):
                    mr["model_index"] = i
                img_data = result.get("image")
                self.finished_signal.emit(img_data, model_results)
            else:
                error_msg = result.get("error", "未知错误") if isinstance(result, dict) else "预测失败"
                self.update_signal.emit(f"预测失败: {error_msg}", None)
                self.finished_signal.emit(None, None)
        except Exception as e:
            self.update_signal.emit(f"预测过程中出现错误: {str(e)}", None)
            self.finished_signal.emit(None, None)


class PredictController:
    def __init__(self, ui_widget, api_client):
        self.ui = ui_widget
        self.api_client = api_client
        self.predict_thread = None
        self._setup_connections()

    def _setup_connections(self):
        self.ui.browse_image_btn.clicked.connect(self.browse_image)
        self.ui.browse_predict_save_btn.clicked.connect(self.browse_save_dir)
        self.ui.add_model_btn.clicked.connect(self.add_model)
        self.ui.remove_model_btn.clicked.connect(self.remove_model)
        self.ui.start_predict_btn.clicked.connect(self.start_prediction)

    def browse_image(self):
        file, _ = QFileDialog.getOpenFileName(self.ui, "选择图片", "", "Image files (*.jpg *.jpeg *.png)")
        if file:
            self.ui.image_input.setText(file)
            self.display_original_image(file)

    def browse_save_dir(self):
        directory = QFileDialog.getExistingDirectory(self.ui, "选择保存目录")
        if directory:
            self.ui.predict_save_input.setText(directory)

    def add_model(self):
        file, _ = QFileDialog.getOpenFileName(self.ui, "选择模型权重", "", "Model files (*.pth)")
        if file:
            item = QListWidgetItem(file)
            self.ui.predict_weights_list.addItem(item)

    def remove_model(self):
        selected_items = self.ui.predict_weights_list.selectedItems()
        for item in selected_items:
            row = self.ui.predict_weights_list.row(item)
            self.ui.predict_weights_list.takeItem(row)

    def display_original_image(self, image_path):
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.ui.original_image_label.width(),
                    self.ui.original_image_label.height(),
                    aspectRatioMode=1
                )
                self.ui.original_image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"显示原图失败: {e}")

    def display_result_image(self, image_data):
        if image_data is None:
            return
        try:
            if isinstance(image_data, str):
                # 尝试作为 base64 编码的图像处理
                try:
                    import base64
                    import io
                    data = base64.b64decode(image_data)
                    image = QImage.fromData(data)
                    if not image.isNull():
                        scaled_image = image.scaled(
                            self.ui.result_image_label.width(),
                            self.ui.result_image_label.height(),
                            aspectRatioMode=1
                        )
                        self.ui.result_image_label.setPixmap(QPixmap.fromImage(scaled_image))
                        return
                except Exception as e:
                    print(f"尝试解析 base64 图像失败: {e}")
                
                # 尝试作为文件路径处理
                pixmap = QPixmap(image_data)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        self.ui.result_image_label.width(),
                        self.ui.result_image_label.height(),
                        aspectRatioMode=1
                    )
                    self.ui.result_image_label.setPixmap(scaled_pixmap)
            elif isinstance(image_data, (bytes, bytearray)):
                image = QImage.fromData(image_data)
                if not image.isNull():
                    scaled_image = image.scaled(
                        self.ui.result_image_label.width(),
                        self.ui.result_image_label.height(),
                        aspectRatioMode=1
                    )
                    self.ui.result_image_label.setPixmap(QPixmap.fromImage(scaled_image))
        except Exception as e:
            print(f"显示结果图失败: {e}")

    def start_prediction(self):
        image_path = self.ui.image_input.text().strip()
        save_dir = self.ui.predict_save_input.text().strip()
        if not image_path:
            QMessageBox.warning(self.ui, "提示", "请选择待预测图片")
            return
        if not self.ui.predict_weights_list.count():
            QMessageBox.warning(self.ui, "提示", "请添加至少一个模型权重")
            return
        if not save_dir:
            QMessageBox.warning(self.ui, "提示", "请选择保存目录")
            return
        model_paths = []
        for i in range(self.ui.predict_weights_list.count()):
            item = self.ui.predict_weights_list.item(i)
            model_paths.append(item.text())
        try:
            box_conf = float(self.ui.box_conf_input.text())
            mask_conf = float(self.ui.mask_conf_input.text())
        except ValueError:
            QMessageBox.warning(self.ui, "提示", "请输入有效的置信度值")
            return
        self.ui.start_predict_btn.setEnabled(False)
        self.ui.predict_log.clear()
        self.predict_thread = PredictThread(
            self.api_client, image_path, model_paths, save_dir, box_conf, mask_conf
        )
        self.predict_thread.update_signal.connect(self.update_log)
        self.predict_thread.finished_signal.connect(self.prediction_finished)
        self.predict_thread.start()

    def update_log(self, message, model_results):
        self.ui.predict_log.append(message)
        if model_results:
            total_instances = 0
            for mr in model_results:
                model_name = mr.get("model", "未知模型")
                num_instances = mr.get("boxes", 0)
                total_instances += num_instances
                
                self.ui.predict_log.append(f"  模型: {model_name}")
                self.ui.predict_log.append(f"  实例数: {num_instances}")
                
                # 尝试从 labels 中获取类别信息
                labels = mr.get("labels", [])
                if labels:
                    # 简单的类别计数
                    label_counts = {}
                    for label in labels:
                        # 假设 label 0 是背景，从 1 开始是实际类别
                        if label > 0:
                            label_name = f"类别 {label}"
                            if label_name not in label_counts:
                                label_counts[label_name] = 0
                            label_counts[label_name] += 1
                    
                    if label_counts:
                        for cls, count in label_counts.items():
                            self.ui.predict_log.append(f"    - {cls}: {count}")
            
            if total_instances > 0:
                self.ui.predict_log.append(f"\n总计: {total_instances} 个实例")

    def prediction_finished(self, image_data, model_results):
        self.ui.start_predict_btn.setEnabled(True)
        if image_data:
            self.display_result_image(image_data)
            # 显示模型结果信息
            self.update_log("预测完成！", model_results)
        else:
            QMessageBox.warning(self.ui, "失败", "预测失败，请查看日志")
