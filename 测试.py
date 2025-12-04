import sys
import csv
import random
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFileDialog, QPushButton, QMessageBox, QTextEdit,
    QProgressBar, QDialog, QScrollArea, QVBoxLayout, QDialogButtonBox)
from PyQt5.QtGui import QFont
from Neuralnetwork import Nnetwork  # 神经网络
from Metalearning import MetaLearning  # 元神经网络训练
from data import support_datas, pattern_datas_2  # 数据库 用于保存用户之前的任务集
from set import support_set_label
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import binary_crossentropy
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QGridLayout, QLabel, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTabWidget, QTextEdit, QVBoxLayout, QWidget  # 新增 QTabWidget 和 QWidget
from sklearn.model_selection import train_test_split


class PatternSelectionWindow(QDialog):
    def __init__(self, patterns, feature_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择感兴趣的模式")
        self.setModal(True)  # 设置为模态对话框
        self.patterns = patterns
        self.feature_names = feature_names
        self.checkboxes = []  # 存储所有的复选框

        # 设置对话框的大小
        self.setFixedSize(800, 500)  # 设置固定大小为 700x500 像素

        # 创建滚动区域
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # 创建容器
        container = QWidget()
        layout = QGridLayout(container)  # 使用网格布局

        # 按照模式长度从小到大排序
        sorted_patterns = sorted(self.patterns, key=lambda x: sum(x))

        # 设置字体样式
        font = QFont("Bahnschrift", 12)  # 字体为 Bahnschrift，大小为 12

        # 直接使用传入的已排序模式（不再二次排序）
        self.sorted_patterns = patterns  # 假设传入的 patterns 已经排序

        # 为每个模式创建一个复选框和标签
        for i, pattern in enumerate(sorted_patterns):
            pattern_list = self.vector_to_list(pattern, self.feature_names)

            # 创建复选框，显示模式列表
            checkbox_text = f"pattern {i + 1}: {pattern_list}"
            checkbox = QCheckBox(checkbox_text, self)
            checkbox.setFont(font)  # 设置字体样式
            self.checkboxes.append(checkbox)

            # 创建标签，显示 feature vector
            feature_vector_label = QLabel(f"(feature vector: {pattern})", self)
            feature_vector_label.setFont(font)  # 设置字体样式
            feature_vector_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # 靠右对齐

            # 将复选框和标签添加到网格布局
            layout.addWidget(checkbox, i, 0)  # 复选框放在第 i 行，第 0 列
            layout.addWidget(feature_vector_label, i, 1)  # 标签放在第 i 行，第 1 列

        # 设置列拉伸比例，使 feature vector 部分整体靠右对齐
        layout.setColumnStretch(0, 1)  # 第 0 列（复选框部分）可以拉伸
        layout.setColumnStretch(1, 0)  # 第 1 列（feature vector 部分）不拉伸

        # 设置滚动区域的部件
        scroll_area.setWidget(container)

        # 创建按钮框
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(button_box)

    def get_selected_labels(self):
        """
        获取用户的选择结果
        :return: 包含每个模式是否被选中的列表（1: 选中, 0: 未选中）
        """
        return [1 if checkbox.isChecked() else 0 for checkbox in self.checkboxes]

    def vector_to_list(self, pattern_vector, feature_names):
        """
        将模式向量转换为对应的列表表示
        :param pattern_vector: 模式向量，例如 [1, 1, 1, 1, 1, 0, 1]
        :param feature_names: 特征名称列表，例如 ['Schools', 'bus stations', 'shopping malls', ...]
        :return: 转换后的列表，例如 ['Schools', 'bus stations', 'shopping malls', 'restaurants', 'police stations', 'parking lots']
        """
        if len(pattern_vector) != len(feature_names):
            raise ValueError("模式向量和特征名称列表的长度不一致。")

        # 提取值为1的特征名称
        selected_features = [feature for feature, value in zip(feature_names, pattern_vector) if value == 1]

        return selected_features


class UserInteraction:
    def __init__(self, data, feature_names):
        """
        初始化 UserInteraction 类
        :param data: 从 CSV 文件中读取的数据（二维列表）
        :param feature_names: 特征名称列表
        """
        self.data = data
        self.feature_names = feature_names
        self.selected_patterns = []
        self.labels = []  # 存储用户的标注结果（1: 感兴趣, 0: 不感兴趣）

    def select_random_patterns(self, num_patterns=20):
        """
        从数据中随机选择指定数量的模式，并按照模式中1的个数排序
        """
        if len(self.data) < num_patterns:
            raise ValueError("数据中的模式数量不足以选择指定的数量。")
        # 随机选择后立即排序
        selected = random.sample(self.data, num_patterns)
        self.selected_patterns = sorted(selected, key=lambda x: sum(x))  # 关键修改：直接存储排序后的模式

    def interact_with_user(self, parent=None):
        """
        与用户交互，确保使用排序后的模式
        """
        if not self.selected_patterns:
            raise ValueError("未选择任何模式，请先调用 select_random_patterns 方法。")

        # 创建窗口时直接传入已排序的 selected_patterns
        selection_window = PatternSelectionWindow(self.selected_patterns, self.feature_names, parent)
        if selection_window.exec_() == QDialog.Accepted:
            self.labels = selection_window.get_selected_labels()
        else:
            self.labels = [0] * len(self.selected_patterns)

    def get_labeled_results(self):
        """
        返回标注结果
        :return: 包含模式和对应标注结果的列表
        """
        return list(zip(self.selected_patterns, self.labels))

    def remove_labeled_patterns(self, data):
        """
        从数据中移除已标注的模式
        :param data: 原始数据列表
        :return: 移除已标注模式后的数据列表
        """
        for pattern in self.selected_patterns:
            if pattern in data:
                data.remove(pattern)
        return data


class PreTrainThread(QThread):
    """用于后台预训练的线程"""
    finished = pyqtSignal(object)  # 信号：预训练完成时发送模型

    def run(self):
        """线程运行时的逻辑"""
        # 加载原始数据
        X, y = MetaLearning.load_data(support_set_label)
        # 对 X 中的每个样本进行标准化
        X_scaled = [StandardScaler().fit_transform(task.reshape(-1, 1)).flatten() for task in X]

        # 获取输入数据的特征维度
        input_dim = len(X_scaled[0])  # 每个样本的特征数量

        # 初始化元模型
        meta_model = MetaLearning.MetaModelWrapper(input_dim, hidden_units=64)

        # 执行 MAML 训练
        for epoch in range(15):  # 假设 epochs=25
            MetaLearning.maml_train(
                meta_model.meta_model,
                X_scaled,
                y,
                num_tasks=10,
                n_shot=20,
                n_query=40,
                k_way=2,
                inner_optimizer=Adam(learning_rate=0.01),
                outer_optimizer=Adam(learning_rate=0.001),
                epochs=1,  # 每次只训练一个 epoch
                inner_steps=5
            )

        # 发送预训练完成的信号
        self.finished.emit(meta_model.meta_model)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口标题
        self.setWindowTitle('Meta-PatternLearner')

        # 设置窗口大小
        self.resize(900, 600)

        # 创建选项卡控件
        self.tab_widget = QTabWidget(self)

        # 创建两个页面
        self.tab_prevalents = QWidget()
        self.tab_predicted = QWidget()

        # 将页面添加到选项卡控件
        self.tab_widget.addTab(self.tab_prevalents, "Prevalents Patterns")
        self.tab_widget.addTab(self.tab_predicted, "Predicted Patterns")

        # 设置 Prevalents 页面的布局
        self.setup_prevalents_tab()
        # 设置 Predicted Patterns 页面的布局
        self.setup_predicted_tab()

        # 创建按钮：打开 CSV 文件
        self.btn_openFile = QPushButton('Open CSV File', self)
        self.btn_openFile.setFixedSize(120, 35)

        # 创建按钮：启动用户交互
        self.btn_interact = QPushButton('Start User Interaction', self)
        self.btn_interact.setFixedSize(150, 35)

        # 创建标签：显示状态信息
        self.label_result = QLabel("Status: Please select a CSV file", self)

        # 创建进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedSize(120, 15)  # 设置进度条大小
        self.progress_bar.setValue(0)  # 初始值为 0

        # 设置自定义字体
        self.set_font(self.btn_openFile)
        self.set_font(self.btn_interact)
        self.set_font(self.label_result)

        # 连接按钮点击事件
        self.btn_openFile.clicked.connect(self.openFile)
        self.btn_interact.clicked.connect(self.start_user_interaction)

        # 设置状态标签的样式
        self.label_result.setFont(QFont("Bahnschrift", 12))  # 设置字体和大小，加粗
        self.label_result.setStyleSheet("""
            QLabel {
                color: #000000;  /* 默认文本颜色为深蓝色 */
                background-color: #FFFFFF;  /* 默认背景色为白色 */
                padding: 1px;  /* 增加内边距 */
                border-radius: 1px;  /* 设置圆角 */
                border: 1px solid #D6DBDF;  /* 设置边框 */
            }
        """)

        # 设置布局
        main_layout = QVBoxLayout()

        # 创建顶部布局，放置按钮
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.btn_openFile)
        top_layout.addWidget(self.btn_interact)
        top_layout.addStretch()  # 添加弹性空间

        # 创建底部布局，放置状态标签和进度条
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.label_result)  # 状态标签放在左下角
        bottom_layout.addStretch()  # 添加弹性空间
        bottom_layout.addWidget(self.progress_bar)  # 进度条放在右下角

        # 将布局添加到主布局
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.tab_widget)  # 添加选项卡控件
        main_layout.addLayout(bottom_layout)
        main_layout.setSpacing(10)  # 设置布局中控件之间的间距

        self.setLayout(main_layout)

        # 属性：存储 CSV 数据
        self.csv_data = None

        # 列表：存储每行的内容
        self.row_data_list = []

        # 列表：用于存储处理后的数据
        self.auxiliary_list = []

        # 字典：存储标注结果
        self.pattern_datas = None

        # 预训练模型
        self.pre_trained_model = None

        # 启动后台预训练
        self.start_pre_training()

    def setup_prevalents_tab(self):
        """设置 Prevalents 页面的布局"""
        layout = QVBoxLayout()

        # 创建文本输出框：显示 Prevalents 信息
        self.text_prevalents = QTextEdit(self.tab_prevalents)
        self.text_prevalents.setReadOnly(True)  # 设置为只读
        self.text_prevalents.setFont(QFont("Bahnschrift", 12))  # 设置字体样式

        # 添加控件到布局
        layout.addWidget(self.text_prevalents)
        self.tab_prevalents.setLayout(layout)

    def setup_predicted_tab(self):
        """设置 Predicted Patterns 页面的布局"""
        layout = QVBoxLayout()

        # 创建文本输出框：显示 Predicted Patterns 信息
        self.text_predicted = QTextEdit(self.tab_predicted)
        self.text_predicted.setReadOnly(True)  # 设置为只读
        self.text_predicted.setFont(QFont("Bahnschrift", 12))  # 设置字体样式

        # 添加控件到布局
        layout.addWidget(self.text_predicted)
        self.tab_predicted.setLayout(layout)

    def start_pre_training(self):
        """启动后台预训练"""
        self.pre_train_thread = PreTrainThread()
        self.pre_train_thread.finished.connect(self.on_pre_train_finished)
        self.pre_train_thread.start()

    def on_pre_train_finished(self, model):
        """预训练完成时的回调函数"""
        self.pre_trained_model = model
        print("预训练完成！")  # 仅在控制台打印，不显示在界面上

    def set_font(self, widget, font_family="Bahnschrift", font_size=10, bold=False, italic=False):
        """设置自定义字体"""
        font = QFont(font_family, font_size)
        font.setBold(bold)
        font.setItalic(italic)
        widget.setFont(font)

    def openFile(self):
        """打开 CSV 文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "./", "CSV Files (*.csv)")

        if file_path:  # 如果用户选择了文件
            try:
                # 使用 csv 模块读取 CSV 文件
                with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    # 读取数据并存储到类属性中
                    self.csv_data = list(reader)
                    # 将每行的内容存储到 row_data_list
                    self.row_data_list = [row for row in self.csv_data]
                    # 显示成功消息
                    self.label_result.setText(f"Status: File read successfully! {len(self.csv_data)} rows read.")
                    # 调用后端处理函数
                    self.process_csv_data()
            except Exception as e:
                # 显示错误消息
                self.label_result.setText(f"Status: Failed to read file: {e}")

    def process_csv_data(self):
        """处理读取的 CSV 数据"""
        if self.csv_data:
            self.auxiliary_list = []
            for row in self.row_data_list:
                # 将每行的值转换为数字（int 或 float）
                numeric_row = []
                for value in row:
                    try:
                        # 先尝试转换为 int
                        numeric_value = int(value)
                    except ValueError:
                        try:
                            # 如果 int 转换失败，尝试转换为 float
                            numeric_value = float(value)
                        except ValueError:
                            # 如果都失败，保留原始值
                            numeric_value = value
                    numeric_row.append(numeric_value)
                self.auxiliary_list.append(numeric_row)

            # 显示频繁模式
            self.display_prevalent_patterns()

        else:
            print("No CSV data was read.")

    def display_prevalent_patterns(self):
        """将频繁模式显示在 Prevalents Patterns 选项卡中"""
        if not self.auxiliary_list:
            return

        feature_names = ['University', 'Hospital', 'Library', 'Restaurants', 'Cinema', 'Pharmacy', 'Museum', 'Theater', 'Hotel', 'Station']

        # 按照模式中1的个数从小到大排序
        prevalent_patterns_sorted = sorted(self.auxiliary_list, key=lambda x: sum(x))

        # 将模式向量转换为字符串
        if prevalent_patterns_sorted:
            pattern_strings = [self.vector_to_string(pattern, feature_names) for pattern in prevalent_patterns_sorted]
            # 使用 HTML 格式增强显示效果，并设置每个 <li> 的 margin-bottom
            result_html = """
                        <style>
                            li {
                                margin-bottom: 10px;  /* 设置每个模式之间的上下间隔为 15px */
                                font-family: Bahnschrift;  /* 设置字体 */
                                font-size: 12pt;  /* 设置字体大小 */
                            }
                        </style>
                        <ul>
                    """
            for s in pattern_strings:
                result_html += f"<li>{s}</li>"
            result_html += "</ul>"

            # 在 Prevalents Patterns 页面显示结果
            self.text_prevalents.setHtml(result_html)
        else:
            self.text_prevalents.clear()

    def start_user_interaction(self):
        """启动用户交互"""
        if not self.auxiliary_list:
            QMessageBox.warning(self, "Error", "No data loaded. Please open a CSV file first.")
            return

        feature_names = ['University', 'Hospital', 'Library', 'Restaurants', 'Cinema', 'Pharmacy', 'Museum', 'Theater', 'Hotel', 'Station']

        # 创建 UserInteraction 对象
        user_interaction = UserInteraction(self.auxiliary_list, feature_names)
        try:
            # 随机选择 10 个模式
            user_interaction.select_random_patterns()

            # 与用户交互
            user_interaction.interact_with_user(self)
            # 获取标注结果
            labeled_results = user_interaction.get_labeled_results()

            # 定义字典 pattern_datas
            self.pattern_datas = {
                'interesting': [],  # 存储感兴趣的模式
                'uninteresting': [],  # 存储不感兴趣的模式
                'label': [[], []]  # 存储标注结果（1: 感兴趣, 0: 不感兴趣）
            }

            # 将标注结果分类存储到 pattern_datas 中
            for pattern, label in labeled_results:
                if label == 1:
                    self.pattern_datas['interesting'].append(pattern)
                    self.pattern_datas['label'][0].append(1)  # 将 1 存储到 label 的第一个列表
                else:
                    self.pattern_datas['uninteresting'].append(pattern)
                    self.pattern_datas['label'][1].append(0)  # 将 0 存储到 label 的第二个列表

            # 从 auxiliary_list 中移除已标注的模式
            self.auxiliary_list = user_interaction.remove_labeled_patterns(self.auxiliary_list)

            # 用户交互完成后，执行后续逻辑
            self.run_post_interaction_logic()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_post_interaction_logic(self):
        """用户交互完成后执行的逻辑"""
        if not self.pre_trained_model:
            QMessageBox.warning(self, "Error", "预训练模型未加载，无法进行预测。")
            return

        # 在测试集上微调模型
        print('-------------------------------------------------------------------', '微调阶段')
        fine_tune_epochs = 3

        # 设置进度条的范围
        self.progress_bar.setRange(0, fine_tune_epochs)  # 微调阶段共 3 个 epoch

        fine_tune_optimizer = Adam(learning_rate=0.01)

        # 加载新数据集 pattern_data_2
        X_new = np.array(self.pattern_datas["interesting"] + self.pattern_datas["uninteresting"])
        y_new = np.array(
            [1] * len(self.pattern_datas["interesting"]) + [0] * len(self.pattern_datas["uninteresting"]))

        # print(X_new)
        # print(y_new)

        X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.5, random_state=42)
        # print(X_train)
        # print(y_train)

        # # 直接使用输入数据中的 interesting 和 uninteresting 部分
        # X = np.array(
        #     pattern_datas_2["interesting"] + pattern_datas_2["uninteresting"])  # 合并 interesting 和 uninteresting
        # y = np.array([1] * len(pattern_datas_2["interesting"]) + [0] * len(pattern_datas_2["uninteresting"]))  # 生成对应的标签
        #
        # # 将数据分为训练集和测试集
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
        #
        # print(len(X_train))

        # 微调模型
        for epoch in range(fine_tune_epochs):
            with tf.GradientTape() as tape:
                logits = self.pre_trained_model(X_train, training=True)
                logits = tf.squeeze(logits, axis=-1)
                loss = tf.reduce_mean(binary_crossentropy(y_train, logits))
            grads = tape.gradient(loss, self.pre_trained_model.trainable_variables)
            fine_tune_optimizer.apply_gradients(zip(grads, self.pre_trained_model.trainable_variables))
            print(f"Fine-tuning Epoch {epoch + 1}/{fine_tune_epochs} | Loss: {loss.numpy():.4f}")
            self.progress_bar.setValue(epoch + 1)  # 更新进度条
            QApplication.processEvents()

        # 评估模型
        test_accuracy, report = MetaLearning.evaluate_model(self.pre_trained_model, X_test, y_test)  # 使用测试集进行评估
        print(f"\n最终模型测试准确率: {test_accuracy:.4f}")
        print("分类报告:\n", report)

        # 使用训练好的模型对 auxiliary_list 中的数据进行预测
        if self.auxiliary_list:
            X_auxiliary = np.array(self.auxiliary_list)
            y_pred = Nnetwork.predict_with_model(self.pre_trained_model, X_auxiliary)

            # 定义特征名称列表
            feature_names = ['University', 'Hospital', 'Library', 'Restaurants', 'Cinema', 'Pharmacy', 'Museum', 'Theater', 'Hotel', 'Station']

            # 获取预测结果为1的模式
            interesting_patterns = [pattern for pattern, pred in zip(self.auxiliary_list, y_pred) if pred == 1]

            # 按照模式中1的个数从小到大排序
            interesting_patterns_sorted = sorted(interesting_patterns, key=lambda x: sum(x))

            # 将模式向量转换为字符串
            if interesting_patterns_sorted:
                pattern_strings = [self.vector_to_string(pattern, feature_names) for pattern in
                                   interesting_patterns_sorted]
                # 使用 HTML 格式增强显示效果，并设置每个 <li> 的 margin-bottom
                result_html = """
                            <style>
                                li {
                                    margin-bottom: 10px;  /* 设置每个模式之间的上下间隔为 15px */
                                    font-family: Bahnschrift;  /* 设置字体 */
                                    font-size: 12pt;  /* 设置字体大小 */
                                }
                            </style>
                            <ul>
                        """
                for s in pattern_strings:
                    result_html += f"<li>{s}</li>"
                result_html += "</ul>"

                # 在 Predicted Patterns 页面显示结果
                self.text_predicted.setHtml(result_html)
                self.label_result.setText("Status: Prediction completed!")  # 状态标签显示简短状态
            else:
                self.text_predicted.clear()
                self.label_result.setText("No interesting patterns found")
        else:
            self.label_result.setText("auxiliary_list is empty and prediction cannot be performed.")

    def vector_to_string(self, pattern_vector, feature_names):
        """
        将模式向量转换为对应的字符串表示
        :param pattern_vector: 模式向量，例如 [1, 1, 1, 1, 1, 0, 1]
        :param feature_names: 特征名称列表，例如 ['Schools', 'bus stations', 'shopping malls', ...]
        :return: 转换后的字符串，例如 "Schools, bus stations, shopping malls, restaurants, police stations, parking lots"
        """
        if len(pattern_vector) != len(feature_names):
            raise ValueError("模式向量和特征名称列表的长度不一致。")

        # 提取值为1的特征名称
        selected_features = [feature for feature, value in zip(feature_names, pattern_vector) if value == 1]

        # 将特征名称拼接成字符串
        return selected_features


if __name__ == "__main__":
    # 创建应用程序对象
    app = QApplication(sys.argv)

    # 创建窗口对象
    window = MainWindow()

    # 显示窗口
    window.show()

    # 运行应用程序的主循环
    app.exec_()

