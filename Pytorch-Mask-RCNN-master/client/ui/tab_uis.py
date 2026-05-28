from PyQt5 import uic
from PyQt5.QtWidgets import QWidget
import os

class TrainTabUI:
    def __init__(self):
        self.ui_path = os.path.join(os.path.dirname(__file__), "train_tab.ui")
        self.widget = None
        self.widgets = {}

    def load(self):
        self.widget = uic.loadUi(self.ui_path)
        self._cache_widgets()
        return self.widget

    def _cache_widgets(self):
        if self.widget is None:
            return
        for name in ['dataset_label', 'dataset_input', 'browse_dataset_btn',
                     'save_label', 'save_input', 'browse_save_btn',
                     'start_train_btn', 'train_progress', 'train_log',
                     'log_group', 'loss_group', 'loss_chart_container',
                     'train_summary', 'summary_group']:
            if hasattr(self.widget, name):
                self.widgets[name] = getattr(self.widget, name)

    def get_widget(self, name):
        return self.widgets.get(name)

class PredictTabUI:
    def __init__(self):
        self.ui_path = os.path.join(os.path.dirname(__file__), "predict_tab.ui")
        self.widget = None
        self.widgets = {}

    def load(self):
        self.widget = uic.loadUi(self.ui_path)
        self._cache_widgets()
        return self.widget

    def _cache_widgets(self):
        if self.widget is None:
            return
        for name in ['image_input', 'browse_image_btn', 'predict_save_input',
                     'browse_predict_save_btn', 'predict_weights_list',
                     'add_model_btn', 'remove_model_btn', 'box_conf_input',
                     'mask_conf_input', 'start_predict_btn',
                     'original_image_label', 'result_image_label', 'predict_log']:
            if hasattr(self.widget, name):
                self.widgets[name] = getattr(self.widget, name)

    def get_widget(self, name):
        return self.widgets.get(name)

class CameraTabUI:
    def __init__(self):
        self.ui_path = os.path.join(os.path.dirname(__file__), "camera_tab.ui")
        self.widget = None
        self.widgets = {}

    def load(self):
        self.widget = uic.loadUi(self.ui_path)
        self._cache_widgets()
        return self.widget

    def _cache_widgets(self):
        if self.widget is None:
            return
        for name in ['camera_selector', 'camera_weights_list', 'add_model_btn',
                     'remove_model_btn', 'camera_box_conf_input', 'camera_mask_conf_input',
                     'start_camera_btn', 'stop_camera_btn', 'camera_interval_combo',
                     'camera_original_label', 'camera_result_label', 'camera_log']:
            if hasattr(self.widget, name):
                self.widgets[name] = getattr(self.widget, name)

    def get_widget(self, name):
        return self.widgets.get(name)

class EvaluateTabUI:
    def __init__(self):
        self.ui_path = os.path.join(os.path.dirname(__file__), "evaluate_tab.ui")
        self.widget = None
        self.widgets = {}

    def load(self):
        self.widget = uic.loadUi(self.ui_path)
        self._cache_widgets()
        return self.widget

    def _cache_widgets(self):
        if self.widget is None:
            return
        for name in ['test_input', 'browse_test_btn', 'weights_input',
                     'browse_weights_btn', 'eval_save_input', 'browse_eval_save_btn',
                     'start_evaluate_btn', 'eval_progress', 'bbox_map50_value',
                     'segm_map50_value', 'ar_max100_value', 'eval_results_table',
                     'export_btn', 'view_history_btn', 'evaluate_log']:
            if hasattr(self.widget, name):
                self.widgets[name] = getattr(self.widget, name)

    def get_widget(self, name):
        return self.widgets.get(name)

class UserTabUI:
    def __init__(self):
        self.ui_path = os.path.join(os.path.dirname(__file__), "user_tab.ui")
        self.widget = None
        self.widgets = {}

    def load(self):
        self.widget = uic.loadUi(self.ui_path)
        self._cache_widgets()
        return self.widget

    def _cache_widgets(self):
        if self.widget is None:
            return
        for name in ['total_users_count', 'admin_count_count', 'normal_users_count',
                     'user_search_input', 'add_user_btn', 'refresh_btn', 'user_table']:
            if hasattr(self.widget, name):
                self.widgets[name] = getattr(self.widget, name)

    def get_widget(self, name):
        return self.widgets.get(name)
