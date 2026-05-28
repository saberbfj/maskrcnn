from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem, QHeaderView, QPushButton, QHBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt

class UserController:
    def __init__(self, ui_widget, api_client):
        self.ui = ui_widget
        self.api_client = api_client
        self.all_users = []
        self._setup_connections()
        self._init_table()
        self.load_users()

    def _setup_connections(self):
        self.ui.add_user_btn.clicked.connect(self.add_user)
        self.ui.refresh_btn.clicked.connect(self.load_users)
        self.ui.user_search_input.textChanged.connect(self.filter_users)

    def _init_table(self):
        header = self.ui.user_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        header.setSectionResizeMode(5, QHeaderView.Fixed)
        self.ui.user_table.setColumnWidth(0, 100)
        self.ui.user_table.setColumnWidth(1, 100)
        self.ui.user_table.setColumnWidth(2, 240)
        self.ui.user_table.setColumnWidth(3, 240)
        self.ui.user_table.setColumnWidth(4, 160)
        self.ui.user_table.setColumnWidth(5, 300)

    def load_users(self):
        result = self.api_client.get_users()
        if result.get("success"):
            users = result.get("users", [])
            self.all_users = users
            total_users = len(users)
            admin_count = sum(1 for user in users if user.get("is_admin"))
            normal_count = total_users - admin_count
            self.ui.total_users_count.setText(str(total_users))
            self.ui.admin_count_count.setText(str(admin_count))
            self.ui.normal_users_count.setText(str(normal_count))
            admin_user = None
            other_users = []
            for user in users:
                if user.get("id") == 1:
                    admin_user = user
                else:
                    other_users.append(user)
            sorted_users = []
            if admin_user:
                sorted_users.append(admin_user)
            sorted_users.extend(other_users)
            self._populate_table(sorted_users)

    def _populate_table(self, users):
        self.ui.user_table.setRowCount(len(users))
        for i, user in enumerate(users):
            user_id = user.get("id")
            username = user.get("username")
            created_at = user.get("created_at", "-")
            last_login = user.get("last_login", "-")
            is_admin = user.get("is_admin")
            self.ui.user_table.setItem(i, 0, QTableWidgetItem(str(user_id)))
            self.ui.user_table.setItem(i, 1, QTableWidgetItem(username))
            self.ui.user_table.setItem(i, 2, QTableWidgetItem(str(created_at) if created_at else "-"))
            self.ui.user_table.setItem(i, 3, QTableWidgetItem(str(last_login) if last_login else "-"))
            self.ui.user_table.setItem(i, 4, QTableWidgetItem("是" if is_admin else "否"))
            self._add_action_buttons(i, user_id, is_admin)

    def _add_action_buttons(self, row, user_id, is_admin):
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(5, 5, 5, 5)
        button_layout.setSpacing(8)
        if user_id != 1:
            edit_btn = QPushButton("编辑")
            edit_btn.setObjectName("successBtn")
            edit_btn.setFixedSize(50, 28)
            edit_btn.setStyleSheet("""
                QPushButton#successBtn {
                    background-color: #2ecc71;
                    color: white;
                    border-radius: 4px;
                    padding: 4px 10px;
                }
                QPushButton#successBtn:hover {
                    background-color: #27ae60;
                }
            """)
            edit_btn.clicked.connect(lambda checked, uid=user_id: self.change_password(uid))
            button_layout.addWidget(edit_btn)
            delete_btn = QPushButton("删除")
            delete_btn.setObjectName("dangerBtn")
            delete_btn.setFixedSize(50, 28)
            delete_btn.setStyleSheet("""
                QPushButton#dangerBtn {
                    background-color: #e74c3c;
                    color: white;
                    border-radius: 4px;
                    padding: 4px 10px;
                }
                QPushButton#dangerBtn:hover {
                    background-color: #c0392b;
                }
            """)
            delete_btn.clicked.connect(lambda checked, uid=user_id: self.delete_user(uid))
            button_layout.addWidget(delete_btn)
        else:
            label = QLabel("（管理员）")
            label.setStyleSheet("color: #9b59b6; font-weight: bold;")
            button_layout.addWidget(label)
        button_layout.addStretch()
        self.ui.user_table.setCellWidget(row, 5, button_widget)

    def filter_users(self, text):
        if not text:
            self._populate_table(self.all_users)
            return
        filtered = [u for u in self.all_users if text.lower() in u.get("username", "").lower()]
        admin_user = None
        other_users = []
        for user in filtered:
            if user.get("id") == 1:
                admin_user = user
            else:
                other_users.append(user)
        sorted_users = []
        if admin_user:
            sorted_users.append(admin_user)
        sorted_users.extend(other_users)
        self._populate_table(sorted_users)

    def add_user(self):
        from PyQt5.QtWidgets import QInputDialog
        username, ok1 = QInputDialog.getText(self.ui, "添加用户", "请输入用户名:")
        if not ok1 or not username:
            return
        password, ok2 = QInputDialog.getText(self.ui, "添加用户", "请输入密码:", echo=2)
        if not ok2 or not password:
            return
        result = self.api_client.register(username, password)
        if result.get("success"):
            QMessageBox.information(self.ui, "成功", "用户添加成功!")
            self.load_users()
        else:
            QMessageBox.warning(self.ui, "失败", f"添加失败: {result.get('error')}")

    def change_password(self, user_id):
        from PyQt5.QtWidgets import QInputDialog
        new_password, ok = QInputDialog.getText(self.ui, "修改密码", "请输入新密码:", echo=2)
        if not ok or not new_password:
            return
        result = self.api_client.change_user_password(user_id, new_password)
        if result.get("success"):
            QMessageBox.information(self.ui, "成功", "密码修改成功!")
        else:
            QMessageBox.warning(self.ui, "失败", f"修改失败: {result.get('error')}")

    def delete_user(self, user_id):
        reply = QMessageBox.question(
            self.ui, "确认", "确定要删除该用户吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        result = self.api_client.delete_user(user_id)
        if result.get("success"):
            QMessageBox.information(self.ui, "成功", "用户删除成功!")
            self.load_users()
        else:
            QMessageBox.warning(self.ui, "失败", f"删除失败: {result.get('error')}")
