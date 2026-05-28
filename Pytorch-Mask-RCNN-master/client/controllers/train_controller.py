from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget
from PyQt5.QtCore import QThread, pyqtSignal
import os

class LossChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        from PyQt5.QtGui import QColor
        from PyQt5.QtCore import Qt, QRectF
        from PyQt5.QtGui import QPainter, QPainterPath, QBrush, QPen
        self.loss_data = {"total": [], "class": [], "box": [], "mask": []}
        self.max_epochs = 50
        self.colors = {
            "total": QColor(255, 99, 132),
            "class": QColor(144, 238, 144),
            "box": QColor(173, 216, 230),
            "mask": QColor(255, 215, 0)
        }
        self.setMinimumSize(300, 300)
        self.setMaximumSize(350, 350)

    def update_data(self, loss_history):
        self.loss_data["total"] = [item.get("total_loss", 0) for item in loss_history]
        self.loss_data["class"] = [item.get("class_loss", 0) for item in loss_history]
        self.loss_data["box"] = [item.get("box_loss", 0) for item in loss_history]
        self.loss_data["mask"] = [item.get("mask_loss", 0) for item in loss_history]
        self.update()

    def paintEvent(self, event):
        try:
            from PyQt5.QtGui import QPainter, QPainterPath, QBrush, QPen, QColor
            from PyQt5.QtCore import Qt, QRectF
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            rect = self.rect()
            padding = 40
            chart_rect = QRectF(padding, padding, rect.width() - padding * 2, rect.height() - padding * 2)
            painter.fillRect(chart_rect, QColor(30, 30, 30))
            painter.setPen(QColor(100, 100, 100))
            painter.drawRect(chart_rect)
            if not self.loss_data["total"]:
                painter.setPen(QColor(200, 200, 200))
                painter.drawText(rect, Qt.AlignCenter, "等待训练开始...")
                return
            all_losses = []
            for key in self.loss_data:
                all_losses.extend(self.loss_data[key])
            if not all_losses:
                return
            max_loss = max(all_losses) if all_losses else 1
            min_loss = min(all_losses) if all_losses else 0
            if max_loss == min_loss:
                max_loss = min_loss + 1
            epoch_count = len(self.loss_data["total"])
            x_scale = chart_rect.width() / max(epoch_count, 1)
            y_scale = chart_rect.height() / (max_loss - min_loss)

            def draw_line(data, color):
                if len(data) < 1:
                    return
                if len(data) == 1:
                    x = int(chart_rect.left() + 0 * x_scale)
                    y = int(chart_rect.bottom() - (data[0] - min_loss) * y_scale)
                    painter.setBrush(QBrush(color))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(x - 3, y - 3, 6, 6)
                    return
                path = QPainterPath()
                for i, value in enumerate(data):
                    x = int(chart_rect.left() + i * x_scale)
                    y = int(chart_rect.bottom() - (value - min_loss) * y_scale)
                    if i == 0:
                        path.moveTo(x, y)
                    else:
                        path.lineTo(x, y)
                pen = QPen(color, 2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)

            draw_line(self.loss_data["total"], self.colors["total"])
            draw_line(self.loss_data["class"], self.colors["class"])
            draw_line(self.loss_data["box"], self.colors["box"])
            draw_line(self.loss_data["mask"], self.colors["mask"])

            painter.setPen(QColor(200, 200, 200))
            painter.setBrush(Qt.NoBrush)
            max_ticks = min(epoch_count, 10)
            step = max(1, epoch_count // max_ticks) if epoch_count > 0 else 1
            for i in range(0, epoch_count + 1, step):
                x = int(chart_rect.left() + i * x_scale)
                painter.drawLine(x, int(chart_rect.bottom()), x, int(chart_rect.bottom()) + 5)
                painter.drawText(x - 10, int(chart_rect.bottom()) + 20, str(i))
            for i in range(5):
                value = min_loss + (max_loss - min_loss) * (4 - i) / 4
                y = int(chart_rect.top() + i * chart_rect.height() / 4)
                painter.drawLine(int(chart_rect.left()) - 5, y, int(chart_rect.left()), y)
                painter.drawText(5, y + 5, f"{value:.2f}")
        except Exception as e:
            print(f"绘图错误: {e}")


class TrainingThread(QThread):
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    loss_signal = pyqtSignal(list)
    finished_signal = pyqtSignal(object)
    task_id_signal = pyqtSignal(str)

    def __init__(self, api_client, dataset_dir, save_dir):
        super().__init__()
        self.api_client = api_client
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.stop_flag = False
        self.logs_shown = []
        self.current_epoch = 0
        self.last_update_time = 0

    def run(self):
        try:
            # 只发送本地开始消息，不重复服务器的消息
            self.update_signal.emit("开始训练任务...")
            result = self.api_client.start_training(self.dataset_dir, self.save_dir)
            if result.get("success"):
                task_id = result.get("task_id")
                self.task_id_signal.emit(task_id)
                # 本地发送任务启动消息
                self.update_signal.emit(f"训练任务已启动，任务ID: {task_id}")
                
                last_epoch = 0
                last_train_completed = False
                last_val_completed = False
                
                while not self.stop_flag:
                    status = self.api_client.get_train_status(task_id)
                    if status.get("success"):
                        task_info = status.get("task", {})
                        status_info = task_info.get("status")
                        progress = task_info.get("progress", 0)
                        logs = task_info.get("logs", [])
                        loss_history = task_info.get("loss_history", [])
                        current_epoch = task_info.get("epoch", 0)
                        train_completed = task_info.get("train_completed", False)
                        val_completed = task_info.get("val_completed", False)
                        
                        # 发送训练和验证完成消息
                        if train_completed and not last_train_completed:
                            self.update_signal.emit(f"第 {current_epoch} 轮训练阶段完成")
                            last_train_completed = True
                        if val_completed and not last_val_completed:
                            self.update_signal.emit(f"第 {current_epoch} 轮验证阶段完成")
                            last_val_completed = True
                        
                        # 处理服务器发送的日志，避免重复
                        new_logs = []
                        for log in logs:
                            # 只过滤重复的日志，不过滤特定类型的消息
                            if log not in self.logs_shown:
                                new_logs.append(log)
                                self.logs_shown.append(log)
                        
                        # 发送新的服务器日志
                        for log in new_logs:
                            self.update_signal.emit(log)
                        
                        self.progress_signal.emit(progress)
                        if loss_history:
                            self.loss_signal.emit(loss_history)
                        
                        if status_info == "completed":
                            self.update_signal.emit("训练完成!")
                            self.finished_signal.emit("训练成功")
                            break
                        elif status_info == "error" or status_info == "stopped":
                            # 本地发送失败消息
                            if status_info == "error":
                                self.update_signal.emit("训练失败!")
                            # 处理错误日志
                            error_logs = []
                            for log in logs:
                                if "错误" in log or "failed" in log.lower():
                                    if log not in self.logs_shown:
                                        error_logs.append(log)
                                        self.logs_shown.append(log)
                            for log in error_logs:
                                self.update_signal.emit(f"错误原因: {log}")
                            self.finished_signal.emit("训练失败" if status_info == "error" else "训练已停止")
                            break
                    self.msleep(30000)
            else:
                self.update_signal.emit(f"启动训练失败: {result.get('error')}")
                self.finished_signal.emit("训练失败")
        except Exception as e:
            self.update_signal.emit(f"训练过程中出现错误: {str(e)}")
            self.finished_signal.emit("训练失败")

    def stop(self):
        self.stop_flag = True


class TrainController:
    def __init__(self, ui_widget, api_client):
        self.ui = ui_widget
        self.api_client = api_client
        self.training_thread = None
        self.loss_chart = LossChartWidget()
        self.current_task_id = None
        self._setup_connections()
        self._init_loss_chart()

    def _setup_connections(self):
        self.ui.browse_dataset_btn.clicked.connect(self.browse_dataset)
        self.ui.browse_save_btn.clicked.connect(self.browse_save_dir)
        self.ui.start_train_btn.clicked.connect(self.start_training)
        self.ui.stop_train_btn.clicked.connect(self.stop_training)

    def _init_loss_chart(self):
        if hasattr(self.ui, 'loss_chart_container'):
            layout = self.ui.loss_chart_container.layout()
            if layout is None:
                from PyQt5.QtWidgets import QVBoxLayout
                layout = QVBoxLayout(self.ui.loss_chart_container)
            layout.addWidget(self.loss_chart)

    def browse_dataset(self):
        directory = QFileDialog.getExistingDirectory(self.ui, "选择数据集目录")
        if directory:
            self.ui.dataset_input.setText(directory)

    def browse_save_dir(self):
        directory = QFileDialog.getExistingDirectory(self.ui, "选择保存目录")
        if directory:
            self.ui.save_input.setText(directory)

    def start_training(self):
        dataset_dir = self.ui.dataset_input.text().strip()
        save_dir = self.ui.save_input.text().strip()
        if not dataset_dir:
            QMessageBox.warning(self.ui, "提示", "请选择数据集目录")
            return
        if not os.path.exists(dataset_dir):
            QMessageBox.warning(self.ui, "提示", "数据集目录不存在")
            return
        if not save_dir:
            QMessageBox.warning(self.ui, "提示", "请选择保存目录")
            return
        os.makedirs(save_dir, exist_ok=True)
        self.ui.start_train_btn.setEnabled(False)
        self.ui.stop_train_btn.setEnabled(True)
        self.ui.dataset_input.setEnabled(False)
        self.ui.save_input.setEnabled(False)
        self.ui.browse_dataset_btn.setEnabled(False)
        self.ui.browse_save_btn.setEnabled(False)
        self.ui.train_log.clear()
        self.ui.train_progress.setValue(0)
        self.training_thread = TrainingThread(self.api_client, dataset_dir, save_dir)
        self.training_thread.update_signal.connect(self.append_log)
        self.training_thread.progress_signal.connect(self.update_progress)
        self.training_thread.loss_signal.connect(self.update_loss_chart)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.task_id_signal.connect(self.save_task_id)
        self.training_thread.start()

    def save_task_id(self, task_id):
        self.current_task_id = task_id

    def stop_training(self):
        if not self.current_task_id:
            QMessageBox.warning(self.ui, "提示", "没有正在进行的训练任务")
            return
        
        reply = QMessageBox.question(
            self.ui, 
            "确认停止", 
            "确定要停止训练吗？停止后不会保存任何模型权重。", 
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            result = self.api_client.stop_training(self.current_task_id)
            if result.get("success"):
                self.append_log("正在停止训练...")
            else:
                QMessageBox.warning(self.ui, "错误", f"停止训练失败: {result.get('error')}")

    def append_log(self, message):
        self.ui.train_log.append(message)

    def update_progress(self, value):
        self.ui.train_progress.setValue(value)

    def update_loss_chart(self, loss_history):
        self.loss_chart.update_data(loss_history)
        if len(loss_history) > 0:
            latest = loss_history[-1]
            summary = f"训练进度: 第 {len(loss_history)} 轮\n"
            summary += f"总损失: {latest.get('total_loss', 0):.4f}\n"
            summary += f"分类损失: {latest.get('class_loss', 0):.4f}\n"
            summary += f"边界框损失: {latest.get('box_loss', 0):.4f}\n"
            summary += f"掩码损失: {latest.get('mask_loss', 0):.4f}"
            self.ui.train_summary.setText(summary)

    def training_finished(self, result):
        self.ui.start_train_btn.setEnabled(True)
        self.ui.stop_train_btn.setEnabled(False)
        self.ui.dataset_input.setEnabled(True)
        self.ui.save_input.setEnabled(True)
        self.ui.browse_dataset_btn.setEnabled(True)
        self.ui.browse_save_btn.setEnabled(True)
        self.current_task_id = None
        QMessageBox.information(self.ui, "提示", result)
