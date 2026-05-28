from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QMessageBox, QWidget, QStackedWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation
from PyQt5.QtGui import QFont

from .api_client import APIClient


class PasswordLineEdit(QWidget):
    """带眼睛按钮的密码输入框"""
    def __init__(self, placeholder="请输入密码", parent=None):
        super().__init__(parent)
        self.init_ui(placeholder)
    
    def init_ui(self, placeholder):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # 密码输入框
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText(placeholder)
        self.password_input.setEchoMode(QLineEdit.Password)
        # 防止按Enter键触发其他控件
        self.password_input.returnPressed.connect(lambda: None)
        layout.addWidget(self.password_input)
        
        # 眼睛按钮
        self.eye_button = QPushButton("👁")
        self.eye_button.setFixedSize(30, 30)
        self.eye_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #ecf0f1;
                border-radius: 15px;
            }
        """)
        self.eye_button.setCheckable(True)
        self.eye_button.clicked.connect(self.toggle_password_visibility)
        layout.addWidget(self.eye_button)
        
        self.setLayout(layout)
    
    def toggle_password_visibility(self):
        """切换密码可见性"""
        if self.eye_button.isChecked():
            self.password_input.setEchoMode(QLineEdit.Normal)
            self.eye_button.setText("👁️")
        else:
            self.password_input.setEchoMode(QLineEdit.Password)
            self.eye_button.setText("👁")
    
    def text(self):
        return self.password_input.text()
    
    def setText(self, text):
        self.password_input.setText(text)
    
    def clear(self):
        self.password_input.clear()
    
    def keyPressEvent(self, event):
        """捕获键盘事件，防止Enter键触发其他控件"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # 捕获Enter键，不做任何处理
            pass
        else:
            # 其他按键正常处理
            super().keyPressEvent(event)


class AuthWindow(QDialog):
    login_success = pyqtSignal(str, int, bool, str)
    
    def __init__(self):
        super().__init__()
        self.api_client = APIClient()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("超市货架商品识别系统")
        self.setGeometry(100, 100, 1000, 800)
        self.setFixedSize(1000, 800)
        
        # 设置样式表
        self.setStyleSheet("""
            * {
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", sans-serif;
            }
            QDialog {
                background-color: #f5f7fa;
            }
            QLabel {
                font-size: 15px;
                font-weight: 500;
                color: #2c3e50;
            }
            QLineEdit {
                padding: 12px 15px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 500;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #3498db;
                outline: none;
            }
            QPushButton {
                padding: 12px 30px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton#loginBtn {
                background-color: #3498db;
                color: white;
            }
            QPushButton#loginBtn:hover {
                background-color: #2980b9;
            }
            QPushButton#registerBtn {
                background-color: #2ecc71;
                color: white;
            }
            QPushButton#registerBtn:hover {
                background-color: #27ae60;
            }
            QPushButton#switchBtn {
                background-color: transparent;
                color: #3498db;
                font-size: 14px;
                padding: 5px 10px;
            }
            QPushButton#switchBtn:hover {
                text-decoration: underline;
            }
            QWidget#formContainer {
                background-color: white;
                border-radius: 16px;
                border: 1px solid #e0e0e0;
                padding: 40px;
            }
            QLabel#titleLabel {
                font-size: 32px;
                font-weight: 700;
                color: #2c3e50;
                margin-bottom: 10px;
            }
            QLabel#subtitleLabel {
                font-size: 18px;
                color: #7f8c8d;
                margin-bottom: 30px;
            }
            QStackedWidget {
                background-color: transparent;
            }
        """)
        
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 顶部装饰条
        top_bar = QWidget()
        top_bar.setFixedHeight(10)
        top_bar.setStyleSheet("background: linear-gradient(90deg, #3498db, #9b59b6);")
        main_layout.addWidget(top_bar)
        
        # 内容区域
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(100, 50, 100, 50)
        
        # 左侧欢迎区域
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout()
        welcome_layout.setAlignment(Qt.AlignCenter)
        welcome_layout.setSpacing(20)
        
        welcome_title = QLabel("欢迎使用")
        welcome_title.setStyleSheet("font-size: 40px; font-weight: 700; color: #2c3e50;")
        welcome_subtitle = QLabel("超市货架商品识别系统")
        welcome_subtitle.setStyleSheet("font-size: 24px; color: #3498db;")
        welcome_desc = QLabel("强大的实例分割工具，支持多种物体识别和分割")
        welcome_desc.setStyleSheet("font-size: 16px; color: #7f8c8d; text-align: center;")
        welcome_desc.setWordWrap(True)
        
        welcome_layout.addWidget(welcome_title)
        welcome_layout.addWidget(welcome_subtitle)
        welcome_layout.addWidget(welcome_desc)
        welcome_widget.setLayout(welcome_layout)
        
        # 右侧表单区域
        form_container = QWidget()
        form_container.setObjectName("formContainer")
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(30)
        
        # 表单标题
        self.title_label = QLabel("登录")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(self.title_label)
        
        subtitle_label = QLabel("请登录或注册以使用系统")
        subtitle_label.setObjectName("subtitleLabel")
        subtitle_label.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(subtitle_label)
        
        # 堆叠窗口，用于切换登录和注册
        self.stacked_widget = QStackedWidget()
        form_layout.addWidget(self.stacked_widget)
        
        # 登录页面
        login_widget = QWidget()
        login_layout = QVBoxLayout()
        login_layout.setSpacing(20)
        
        # 用户名输入
        username_layout = QVBoxLayout()
        username_label = QLabel("用户名")
        self.login_username = QLineEdit()
        self.login_username.setPlaceholderText("请输入用户名")
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.login_username)
        login_layout.addLayout(username_layout)
        
        # 密码输入
        password_layout = QVBoxLayout()
        password_label = QLabel("密码")
        self.login_password = PasswordLineEdit("请输入密码")
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.login_password)
        login_layout.addLayout(password_layout)
        
        # 登录按钮
        self.login_button = QPushButton("🔐 登录")
        self.login_button.setObjectName("loginBtn")
        self.login_button.clicked.connect(self.handle_login)
        login_layout.addWidget(self.login_button)
        
        # 切换到注册
        switch_layout = QHBoxLayout()
        switch_layout.setAlignment(Qt.AlignCenter)
        switch_label = QLabel("还没有账号？")
        self.switch_to_register_button = QPushButton("立即注册")
        self.switch_to_register_button.setObjectName("switchBtn")
        self.switch_to_register_button.clicked.connect(self.switch_to_register_with_animation)
        switch_layout.addWidget(switch_label)
        switch_layout.addWidget(self.switch_to_register_button)
        login_layout.addLayout(switch_layout)
        
        login_widget.setLayout(login_layout)
        self.stacked_widget.addWidget(login_widget)
        
        # 注册页面
        register_widget = QWidget()
        register_layout = QVBoxLayout()
        register_layout.setSpacing(20)
        
        # 注册用户名输入
        reg_username_layout = QVBoxLayout()
        reg_username_label = QLabel("用户名")
        self.register_username = QLineEdit()
        self.register_username.setPlaceholderText("请输入用户名")
        reg_username_layout.addWidget(reg_username_label)
        reg_username_layout.addWidget(self.register_username)
        register_layout.addLayout(reg_username_layout)
        
        # 注册密码输入
        reg_password_layout = QVBoxLayout()
        reg_password_label = QLabel("密码")
        self.register_password = PasswordLineEdit("请输入密码（至少6位）")
        reg_password_layout.addWidget(reg_password_label)
        reg_password_layout.addWidget(self.register_password)
        register_layout.addLayout(reg_password_layout)
        
        # 注册按钮
        self.register_button = QPushButton("📝 注册")
        self.register_button.setObjectName("registerBtn")
        self.register_button.clicked.connect(self.handle_register)
        register_layout.addWidget(self.register_button)
        
        # 切换到登录
        reg_switch_layout = QHBoxLayout()
        reg_switch_layout.setAlignment(Qt.AlignCenter)
        reg_switch_label = QLabel("已有账号？")
        self.switch_to_login_button = QPushButton("立即登录")
        self.switch_to_login_button.setObjectName("switchBtn")
        self.switch_to_login_button.clicked.connect(self.switch_to_login_with_animation)
        reg_switch_layout.addWidget(reg_switch_label)
        reg_switch_layout.addWidget(self.switch_to_login_button)
        register_layout.addLayout(reg_switch_layout)
        
        register_widget.setLayout(register_layout)
        self.stacked_widget.addWidget(register_widget)
        
        content_layout.addWidget(welcome_widget, 1)
        content_layout.addWidget(form_container, 1)
        main_layout.addLayout(content_layout)
        
        self.setLayout(main_layout)
    
    def switch_to_register_with_animation(self):
        """切换到注册表单"""
        self.stacked_widget.setCurrentIndex(1)
        self.title_label.setText("注册")
    
    def switch_to_login_with_animation(self):
        """切换到登录表单"""
        self.stacked_widget.setCurrentIndex(0)
        self.title_label.setText("登录")
    
    def handle_login(self):
        username = self.login_username.text().strip()
        password = self.login_password.text().strip()
        
        if not username or not password:
            QMessageBox.warning(self, "警告", "用户名和密码不能为空")
            return
        
        result = self.api_client.login(username, password)
        print(f"登录结果: {result}")
        if result.get("success"):
            user_id = result.get("user_id")
            is_admin = result.get("is_admin")
            print(f"准备发出登录成功信号: username={username}, user_id={user_id}, is_admin={is_admin}")
            self.login_success.emit(username, user_id, is_admin, password)
            print("信号已发出")
            self.close()
            print("登录窗口已关闭")
        else:
            QMessageBox.warning(self, "登录失败", result.get("error"))
    
    def handle_register(self):
        username = self.register_username.text().strip()
        password = self.register_password.text().strip()
        
        if not username or not password:
            QMessageBox.warning(self, "警告", "用户名和密码不能为空")
            return
        
        if len(username) < 3:
            QMessageBox.warning(self, "警告", "用户名至少需要3个字符")
            return
        
        if len(password) < 6:
            QMessageBox.warning(self, "警告", "密码至少需要6个字符")
            return
        
        result = self.api_client.register(username, password)
        if result.get("success"):
            QMessageBox.information(self, "注册成功", result.get("message"))
            self.stacked_widget.setCurrentIndex(0)
        else:
            QMessageBox.warning(self, "注册失败", result.get("error"))