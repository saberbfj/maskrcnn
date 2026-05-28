import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.api_client import APIClient
from client.auth_window import AuthWindow
from client.ui.tab_uis import TrainTabUI, PredictTabUI, CameraTabUI, EvaluateTabUI, UserTabUI
from client.controllers import TrainController, PredictController, CameraController, EvaluateController, UserController


class MainWindow(QMainWindow):
    def __init__(self, username, user_id, is_admin, password):
        super().__init__()
        print(f"MainWindow.__init__ called with: username={username}, user_id={user_id}, is_admin={is_admin}")
        self.username = username
        self.user_id = user_id
        self.is_admin = is_admin
        self.api_client = APIClient()
        self.api_client.user_info = {
            "username": username,
            "user_id": user_id,
            "is_admin": is_admin,
            "password": password
        }
        self.controllers = {}
        print("开始初始化UI...")
        self.init_ui()
        print("UI初始化完成")

    def init_ui(self):
        self.setWindowTitle(f"超市货架商品识别系统 - 用户: {self.username}")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            * {
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", sans-serif;
            }
            QMainWindow {
                background-color: #f5f7fa;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-color: #3498db;
                color: #3498db;
            }
            QPushButton {
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton#primaryBtn {
                background-color: #3498db;
                color: white;
            }
            QPushButton#primaryBtn:hover {
                background-color: #2980b9;
            }
            QPushButton#primaryBtn:pressed {
                background-color: #2471a3;
            }
            QPushButton#successBtn {
                background-color: #2ecc71;
                color: white;
            }
            QPushButton#successBtn:hover {
                background-color: #27ae60;
            }
            QPushButton#dangerBtn {
                background-color: #e74c3c;
                color: white;
            }
            QPushButton#dangerBtn:hover {
                background-color: #c0392b;
            }
            QLineEdit, QComboBox {
                padding: 8px 12px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                font-size: 14px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #3498db;
                outline: none;
            }
            QTextEdit {
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                font-size: 14px;
            }
            QProgressBar {
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #f0f0f0;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 4px;
            }
            QListWidget {
                border: 2px solid #e0e0e0;
                border-radius: 6px;
            }
            QTableWidget {
                border: 2px solid #e0e0e0;
                border-radius: 6px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::header {
                background-color: #f0f0f0;
                font-weight: 600;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        info_bar = QHBoxLayout()
        info_label = QLabel(f"欢迎, {self.username} {'(管理员)' if self.is_admin else ''}")
        info_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #2c3e50;")
        logout_button = QPushButton("退出登录")
        logout_button.setObjectName("dangerBtn")
        logout_button.clicked.connect(self.logout)
        info_bar.addWidget(info_label)
        info_bar.addStretch()
        info_bar.addWidget(logout_button)
        main_layout.addLayout(info_bar)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        print("init_ui: 开始创建Tab组件...")
        train_ui = TrainTabUI()
        print("init_ui: TrainTabUI 已创建")
        train_tab = train_ui.load()
        print("init_ui: TrainTabUI.load() 完成")
        self.controllers['train'] = TrainController(train_tab, self.api_client)
        print("init_ui: TrainController 已创建")
        self.tab_widget.addTab(train_tab, "模型训练")
        print("init_ui: 训练Tab已添加")

        predict_ui = PredictTabUI()
        predict_tab = predict_ui.load()
        self.controllers['predict'] = PredictController(predict_tab, self.api_client)
        self.tab_widget.addTab(predict_tab, "图片预测")

        camera_ui = CameraTabUI()
        camera_tab = camera_ui.load()
        self.controllers['camera'] = CameraController(camera_tab, self.api_client)
        self.tab_widget.addTab(camera_tab, "摄像头预测")

        evaluate_ui = EvaluateTabUI()
        evaluate_tab = evaluate_ui.load()
        self.controllers['evaluate'] = EvaluateController(evaluate_tab, self.api_client)
        self.tab_widget.addTab(evaluate_tab, "模型评估")

        if self.is_admin:
            user_ui = UserTabUI()
            user_tab = user_ui.load()
            self.controllers['user'] = UserController(user_tab, self.api_client)
            self.tab_widget.addTab(user_tab, "用户管理")
        print("init_ui: 所有Tab已添加完成")

    def logout(self):
        if hasattr(self, 'controllers') and 'camera' in self.controllers:
            self.controllers['camera'].cleanup()
        self.close()
        self.auth_window = AuthWindow()
        self.auth_window.show()


def main():
    app = QApplication(sys.argv)
    auth_window = AuthWindow()
    main_window = None

    def on_login_success(username, user_id, is_admin, password):
        nonlocal main_window
        print(f"收到登录成功信号: username={username}, user_id={user_id}, is_admin={is_admin}")
        try:
            print("开始创建主窗口...")
            main_window = MainWindow(username, user_id, is_admin, password)
            print("主窗口已创建，设置auth_window引用")
            main_window.auth_window = auth_window
            print("调用main_window.show()...")
            main_window.show()
            print("main_window.show() 已调用")
        except Exception as e:
            print(f"创建主窗口失败: {e}")
            import traceback
            traceback.print_exc()

    auth_window.login_success.connect(on_login_success)
    auth_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
