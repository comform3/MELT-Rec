import sys
import csv
import random
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QPushButton, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtGui import QFont


class UserInteraction:
    def __init__(self, data):
        """
        初始化 UserInteraction 类
        :param data: 从 CSV 文件中读取的数据（二维列表）
        """
        self.data = data
        self.selected_patterns = []
        self.labels = []  # 存储用户的标注结果（1: 感兴趣, 0: 不感兴趣）

    def select_random_patterns(self, num_patterns=10):
        """
        从数据中随机选择指定数量的模式
        :param num_patterns: 要选择的模式数量（默认为 10）
        """
        if len(self.data) < num_patterns:
            raise ValueError("数据中的模式数量不足以选择指定的数量。")
        self.selected_patterns = random.sample(self.data, num_patterns)

    def interact_with_user(self):
        """
        与用户交互，让用户标注感兴趣或不感兴趣
        """
        if not self.selected_patterns:
            raise ValueError("未选择任何模式，请先调用 select_random_patterns 方法。")

        for i, pattern in enumerate(self.selected_patterns):
            # 将模式转换为字符串以便显示
            pattern_str = ", ".join(map(str, pattern))
            # 弹出对话框，让用户选择是否感兴趣
            reply = QMessageBox.question(
                None,
                f"模式 {i + 1}/{len(self.selected_patterns)}",
                f"您对以下模式感兴趣吗？\n\n{pattern_str}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            # 根据用户的选择记录标注结果
            if reply == QMessageBox.Yes:
                self.labels.append(1)  # 感兴趣
            else:
                self.labels.append(0)  # 不感兴趣

    def get_labeled_results(self):
        """
        返回标注结果
        :return: 包含模式和对应标注结果的列表
        """
        return list(zip(self.selected_patterns, self.labels))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Set the window title
        self.setWindowTitle('Meta-PatternLearner')

        # Set the window size
        self.resize(1000, 700)

        # Create a button to open CSV file
        self.btn_openFile = QPushButton('Open CSV File', self)
        self.btn_openFile.setFixedSize(120, 35)

        # Create a button to start user interaction
        self.btn_interact = QPushButton('Start User Interaction', self)
        self.btn_interact.setFixedSize(150, 35)

        # Create a label to display the result
        self.label_result = QLabel("Please select a CSV file", self)

        # Set custom font for the button and label
        self.set_font(self.btn_openFile)
        self.set_font(self.btn_interact)
        self.set_font(self.label_result)

        # Connect the button click events to methods
        self.btn_openFile.clicked.connect(self.openFile)
        self.btn_interact.clicked.connect(self.start_user_interaction)

        # Set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.btn_openFile)
        layout.addWidget(self.btn_interact)
        layout.addWidget(self.label_result)
        self.setLayout(layout)

        # Attribute to store CSV data
        self.csv_data = None

        # List to store each row's content
        self.row_data_list = []

        # Empty list for future use
        self.auxiliary_list = []

    def set_font(self, widget, font_family="Bahnschrift", font_size=10, bold=False, italic=False):
        """Helper function to set custom font for a widget"""
        font = QFont(font_family, font_size)
        font.setBold(bold)
        font.setItalic(italic)
        widget.setFont(font)

    def openFile(self):
        # Open a file dialog to select a CSV file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "./", "CSV Files (*.csv)")

        if file_path:  # If the user selected a file
            try:
                # Use the csv module to read the CSV file
                with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    # Read the data and store it in the class attribute
                    self.csv_data = list(reader)
                    # Save each row's content to row_data_list
                    self.row_data_list = [row for row in self.csv_data]
                    # Display a success message
                    self.label_result.setText(f"File read successfully! {len(self.csv_data)} rows read.")
                    # Call the backend processing function
                    self.process_csv_data()
            except Exception as e:
                # Display an error message
                self.label_result.setText(f"Failed to read file: {e}")

    def process_csv_data(self):
        """Backend processing function to handle the read CSV data"""
        if self.csv_data:
            self.auxiliary_list = []
            for row in self.row_data_list:
                # Convert each value in the row to a number (int or float)
                numeric_row = []
                for value in row:
                    try:
                        # Try converting to int first
                        numeric_value = int(value)
                    except ValueError:
                        try:
                            # If int conversion fails, try converting to float
                            numeric_value = float(value)
                        except ValueError:
                            # If both conversions fail, keep the original value
                            numeric_value = value
                    numeric_row.append(numeric_value)
                self.auxiliary_list.append(numeric_row)
        else:
            print("No CSV data was read.")

    def start_user_interaction(self):
        """启动用户交互"""
        if not self.auxiliary_list:
            QMessageBox.warning(self, "Error", "No data loaded. Please open a CSV file first.")
            return

        # 创建 UserInteraction 对象
        user_interaction = UserInteraction(self.auxiliary_list)
        try:
            # 随机选择 10 个模式
            user_interaction.select_random_patterns()
            # 与用户交互
            user_interaction.interact_with_user()
            # 获取标注结果
            labeled_results = user_interaction.get_labeled_results()
            print(labeled_results)
            # 打印或处理标注结果
            print("Labeled Results:")
            for pattern, label in labeled_results:
                print(f"Pattern: {pattern}, Label: {'Interested' if label else 'Not Interested'}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# Create an application object
app = QApplication(sys.argv)

# Create a window object
window = MainWindow()

# Show the window
window.show()

# Run the application's main loop
app.exec_()