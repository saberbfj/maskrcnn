from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5.QtCore import QThread, pyqtSignal

class EvaluateThread(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object, str)

    def __init__(self, api_client, test_dir, weights_path, save_dir):
        super().__init__()
        self.api_client = api_client
        self.test_dir = test_dir
        self.weights_path = weights_path
        self.save_dir = save_dir

    def run(self):
        try:
            self.update_signal.emit("开始评估...")
            result = self.api_client.evaluate(
                self.test_dir,
                self.weights_path,
                self.save_dir
            )
            if result.get("success"):
                self.update_signal.emit("评估完成!")
                self.finished_signal.emit(result, result.get("save_path"))
            else:
                self.update_signal.emit(f"评估失败: {result.get('error')}")
                self.finished_signal.emit(None, None)
        except Exception as e:
            self.update_signal.emit(f"评估过程中出现错误: {str(e)}")
            self.finished_signal.emit(None, None)


class EvaluateController:
    def __init__(self, ui_widget, api_client):
        self.ui = ui_widget
        self.api_client = api_client
        self.evaluate_thread = None
        self.current_result = None
        self._setup_connections()
        self._init_table()

    def _setup_connections(self):
        self.ui.browse_test_btn.clicked.connect(self.browse_test_dir)
        self.ui.browse_weights_btn.clicked.connect(self.browse_weights)
        self.ui.browse_eval_save_btn.clicked.connect(self.browse_save_dir)
        self.ui.start_evaluate_btn.clicked.connect(self.start_evaluation)
        # 移除导出结果和查看历史功能
        self.ui.export_btn.setVisible(False)
        self.ui.view_history_btn.setVisible(False)

    def _init_table(self):
        self.ui.eval_results_table.setRowCount(9)
        metrics = [
            ("mAP50-95", "--", "--", "COCO-style mAP averaged over IoU thresholds"),
            ("mAP50", "--", "--", "mAP at IoU=0.50"),
            ("mAP75", "--", "--", "mAP at IoU=0.75"),
            ("AP Small", "--", "--", "AP for small objects (area < 32²)"),
            ("AP Medium", "--", "--", "AP for medium objects (32² < area < 96²)"),
            ("AP Large", "--", "--", "AP for large objects (area > 96²)"),
            ("AR max1", "--", "--", "AR with max 1 detection per image"),
            ("AR max10", "--", "--", "AR with max 10 detections per image"),
            ("AR max100", "--", "--", "AR with max 100 detections per image"),
        ]
        for i, (name, bbox_val, segm_val, desc) in enumerate(metrics):
            self.ui.eval_results_table.setItem(i, 0, QTableWidgetItem(name))
            self.ui.eval_results_table.setItem(i, 1, QTableWidgetItem(bbox_val))
            self.ui.eval_results_table.setItem(i, 2, QTableWidgetItem(segm_val))
            self.ui.eval_results_table.setItem(i, 3, QTableWidgetItem(desc))

    def browse_test_dir(self):
        directory = QFileDialog.getExistingDirectory(self.ui, "选择测试集目录")
        if directory:
            self.ui.test_input.setText(directory)

    def browse_weights(self):
        file, _ = QFileDialog.getOpenFileName(self.ui, "选择模型权重", "", "Model files (*.pth)")
        if file:
            self.ui.weights_input.setText(file)

    def browse_save_dir(self):
        directory = QFileDialog.getExistingDirectory(self.ui, "选择保存目录")
        if directory:
            self.ui.eval_save_input.setText(directory)

    def start_evaluation(self):
        test_dir = self.ui.test_input.text().strip()
        weights_path = self.ui.weights_input.text().strip()
        save_dir = self.ui.eval_save_input.text().strip()
        if not test_dir:
            QMessageBox.warning(self.ui, "提示", "请选择测试集目录")
            return
        if not weights_path:
            QMessageBox.warning(self.ui, "提示", "请选择模型权重文件")
            return
        if not save_dir:
            QMessageBox.warning(self.ui, "提示", "请选择保存目录")
            return
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.ui.start_evaluate_btn.setEnabled(False)
        self.ui.evaluate_log.clear()
        self.evaluate_thread = EvaluateThread(self.api_client, test_dir, weights_path, save_dir)
        self.evaluate_thread.update_signal.connect(self.update_log)
        self.evaluate_thread.finished_signal.connect(self.evaluation_finished)
        self.evaluate_thread.start()

    def update_log(self, message):
        self.ui.evaluate_log.append(message)

    def evaluation_finished(self, result, save_path):
        self.ui.start_evaluate_btn.setEnabled(True)
        self.current_result = result
        if result:
            self.display_results(result)
            QMessageBox.information(self.ui, "完成", "评估完成!")
        else:
            QMessageBox.warning(self.ui, "失败", "评估失败，请查看日志")

    def display_results(self, result):
        data = result.get("data", {})
        if data:
            self.ui.bbox_map50_value.setText(f"{data.get('bbox_map50', 0):.4f}")
            self.ui.segm_map50_value.setText(f"{data.get('segm_map50', 0):.4f}")
            self.ui.ar_max100_value.setText(f"{data.get('bbox_ar_max100', 0):.4f}")
            metrics = [
                ("mAP50-95", f"{data.get('bbox_map50_95', 0):.4f}", f"{data.get('segm_map50_95', 0):.4f}"),
                ("mAP50", f"{data.get('bbox_map50', 0):.4f}", f"{data.get('segm_map50', 0):.4f}"),
                ("mAP75", f"{data.get('bbox_map75', 0):.4f}", f"{data.get('segm_map75', 0):.4f}"),
                ("AP Small", f"{data.get('bbox_ap_small', 0):.4f}", f"{data.get('segm_ap_small', 0):.4f}"),
                ("AP Medium", f"{data.get('bbox_ap_medium', 0):.4f}", f"{data.get('segm_ap_medium', 0):.4f}"),
                ("AP Large", f"{data.get('bbox_ap_large', 0):.4f}", f"{data.get('segm_ap_large', 0):.4f}"),
                ("AR max1", f"{data.get('bbox_ar_max1', 0):.4f}", f"{data.get('segm_ar_max1', 0):.4f}"),
                ("AR max10", f"{data.get('bbox_ar_max10', 0):.4f}", f"{data.get('segm_ar_max10', 0):.4f}"),
                ("AR max100", f"{data.get('bbox_ar_max100', 0):.4f}", f"{data.get('segm_ar_max100', 0):.4f}"),
            ]
            for i, (name, bbox_val, segm_val) in enumerate(metrics):
                self.ui.eval_results_table.item(i, 1).setText(bbox_val)
                self.ui.eval_results_table.item(i, 2).setText(segm_val)
