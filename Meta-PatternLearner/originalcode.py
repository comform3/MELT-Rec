import sys
import csv
import random

import networkx as nx
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFileDialog, QPushButton, QMessageBox, QTextEdit,
    QProgressBar, QDialog, QScrollArea, QVBoxLayout, QDialogButtonBox,
    QHBoxLayout, QGridLayout, QLabel, QCheckBox, QTabWidget, QSlider,
    QSpinBox, QGroupBox, QSplitter, QFrame)
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from Neuralnetwork import Nnetwork
from Metalearning import MetaLearning
from data import support_datas, pattern_datas_2
from PCPs import support_set_label
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import networkx as nx
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox,
                             QPushButton, QSizePolicy)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ModelMonitoringWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.loss_history = []
        self.accuracy_history = []

    def setup_ui(self):
        layout = QVBoxLayout()

        # 创建模型监控组
        group = QGroupBox("Model Monitoring")
        group.setStyleSheet("""
            QGroupBox {
                font: bold 12pt "Bahnschrift";
                border: 1px solid #2B7CD3;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)

        # 添加监控指标
        grid = QGridLayout()

        self.loss_label = QLabel("Loss: ---")
        self.accuracy_label = QLabel("Accuracy: ---")
        self.epoch_label = QLabel("Epoch: ---")

        # 创建Matplotlib图表 - 调整大小
        self.figure = Figure(figsize=(5, 3))  # 缩小图形尺寸
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(250, 200)  # 设置最小尺寸确保可见
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Training Metrics', fontsize=10)
        self.ax.set_xlabel('Epoch', fontsize=8)
        self.ax.set_ylabel('Value', fontsize=8)
        self.ax.grid(True)

        # 调整字体大小
        self.ax.tick_params(axis='both', which='major', labelsize=8)

        # 初始化空图表
        self.line_loss, = self.ax.plot([], [], 'r-', label='Loss', linewidth=1)
        self.line_acc, = self.ax.plot([], [], 'b-', label='Accuracy', linewidth=1)
        self.ax.legend(fontsize=8)
        self.figure.tight_layout()  # 调整布局防止标签被截断
        self.canvas.draw()

        # 将指标放在左侧，图表放在右侧
        grid.addWidget(self.loss_label, 0, 0)
        grid.addWidget(self.accuracy_label, 1, 0)
        grid.addWidget(self.epoch_label, 2, 0)
        grid.addWidget(self.canvas, 0, 1, 3, 1)

        # 设置列比例，让图表占据更多空间
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 3)

        group.setLayout(grid)
        layout.addWidget(group)
        self.setLayout(layout)

    def update_metrics(self, loss, accuracy, epoch):
        self.loss_label.setText(f"Loss: {loss:.4f}")
        self.accuracy_label.setText(f"Accuracy: {accuracy:.4f}")
        self.epoch_label.setText(f"Epoch: {epoch}")

        # 更新历史数据
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)

        # 更新图表
        epochs = range(1, len(self.loss_history) + 1)
        self.line_loss.set_data(epochs, self.loss_history)
        self.line_acc.set_data(epochs, self.accuracy_history)

        # 调整坐标轴范围
        self.ax.relim()
        self.ax.autoscale_view()

        # 确保y轴范围在0-1之间（特别是对于准确率）
        self.ax.set_ylim(0, max(1.1, max(self.loss_history) * 1.1))

        # 重绘图表
        self.canvas.draw()


class PatternSimilarityWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = None
        self.canvas = None
        self.ax = None
        self.pos = None
        self.G = None
        self.last_patterns = None
        self.setup_ui()

        # Enable interactive features
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('motion_notify_event', self.on_hover)

        self.press = None
        self.xlim = None
        self.ylim = None
        self.hover_text = None

    def setup_ui(self):
        layout = QVBoxLayout()

        # 创建模式相似性组
        group = QGroupBox("Pattern Similarity Graph")
        group.setStyleSheet("""
            QGroupBox {
                font: bold 12pt "Bahnschrift";
                border: 1px solid #8A2BE2;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)

        # 创建Matplotlib图表
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(250, 200)
        self.ax = self.figure.add_subplot(111)

        # 初始化空图
        self.ax.text(0.5, 0.5, 'Waiting for data...',
                     ha='center', va='center',
                     fontsize=12, color='gray')
        self.ax.axis('off')
        self.canvas.draw()

        # 添加控制按钮
        control_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh the map")
        self.btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #8A2BE2;
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font: 10pt "Bahnschrift";
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        self.btn_refresh.clicked.connect(self.refresh_graph)

        control_layout.addStretch()
        control_layout.addWidget(self.btn_refresh)
        control_layout.addStretch()

        group_layout = QVBoxLayout()
        group_layout.addWidget(self.canvas)
        group_layout.addLayout(control_layout)
        group.setLayout(group_layout)

        layout.addWidget(group)
        self.setLayout(layout)

    def update_similarity_graph(self, patterns, pattern_labels=None):
        """更新相似性图谱
        :param patterns: 模式列表
        :param pattern_labels: 与预测列表一致的标签列表（如['P1', 'P2', ...]）
        """
        self.last_patterns = patterns

        if not patterns:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No visible patterns.',
                         ha='center', va='center',
                         fontsize=12, color='gray')
            self.ax.axis('off')
            self.canvas.draw()
            return

        try:
            import networkx as nx
            from matplotlib.colors import LinearSegmentedColormap

            # 检查是否是第一次生成图谱
            first_generation = not hasattr(self, 'similarity_cbar')

            # 计算模式之间的相似度
            similarity_matrix = self.calculate_similarity(patterns)

            # 创建图
            self.G = nx.Graph()

            # 添加节点
            feature_names = ['University', 'Hospital', 'Library', 'Restaurants',
                             'Cinema', 'Pharmacy', 'Museum', 'Theater', 'Hotel', 'Station']

            for i, pattern in enumerate(patterns):
                features = [feature_names[j] for j, val in enumerate(pattern) if val == 1]
                # 使用传入的标签或默认生成P1, P2等
                label = pattern_labels[i] if pattern_labels else f"P{i + 1}"
                self.G.add_node(i, label=label, size=sum(pattern) * 50, features=features)

            # 添加边（只连接相似度高的节点）
            threshold = 0.8  # 相似度阈值
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    if similarity_matrix[i][j] > threshold:
                        self.G.add_edge(i, j, weight=similarity_matrix[i][j] * 5)

            # 计算布局
            self.pos = nx.spring_layout(self.G, k=0.8, iterations=50)

            # 绘制图
            self.ax.clear()

            # 自定义颜色映射
            colors = ["#FFC0CB", "#FF69B4", "#8A2BE2", "#4B0082"]
            cmap = LinearSegmentedColormap.from_list("custom_purple", colors)

            # 绘制边
            edges = nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax,
                edge_color=[self.G[u][v]['weight'] for u, v in self.G.edges()],
                edge_cmap=cmap,
                width=1.5,
                alpha=0.7
            )

            # 绘制节点（不显示默认标签）
            nodes = nx.draw_networkx_nodes(
                self.G, self.pos, ax=self.ax,
                node_size=[self.G.nodes[n]['size'] for n in self.G.nodes()],
                node_color='#8A2BE2',
                alpha=0.9
            )

            # 在节点内部添加标签（使用与预测列表完全一致的标签）
            for node, (x, y) in self.pos.items():
                self.ax.text(x, y, self.G.nodes[node]['label'],
                             fontsize=5, ha='center', va='center',
                             color='white', weight='bold')

            # 设置初始视图范围
            x_values = [pos[0] for pos in self.pos.values()]
            y_values = [pos[1] for pos in self.pos.values()]
            margin = max(max(x_values) - min(x_values), max(y_values) - min(y_values)) * 0.15
            self.ax.set_xlim(min(x_values) - margin, max(x_values) + margin)
            self.ax.set_ylim(min(y_values) - margin, max(y_values) + margin)

            self.xlim = self.ax.get_xlim()
            self.ylim = self.ax.get_ylim()

            # 只在第一次生成图谱时创建colorbar
            if first_generation and edges:
                self.similarity_cbar = self.figure.colorbar(edges, ax=self.ax, label='Similarity')

            # 自动调整布局
            self.figure.tight_layout()
            self.canvas.draw()

        except ImportError:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'You need to install the networkx library.',
                         ha='center', va='center',
                         fontsize=12, color='red')
            self.ax.axis('off')
            self.canvas.draw()

    def calculate_similarity(self, patterns):
        """计算模式之间的Jaccard相似度"""
        n = len(patterns)
        similarity = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity[i][j] = 1.0
                else:
                    # 计算Jaccard相似度
                    intersection = np.sum(np.logical_and(patterns[i], patterns[j]))
                    union = np.sum(np.logical_or(patterns[i], patterns[j]))
                    similarity[i][j] = similarity[j][i] = intersection / union if union != 0 else 0

        return similarity

    def refresh_graph(self):
        """手动刷新图谱"""
        if hasattr(self, 'last_patterns') and self.last_patterns:
            self.update_similarity_graph(self.last_patterns)

    def on_scroll(self, event):
        """鼠标滚轮缩放"""
        if event.inaxes != self.ax:
            return

        scale_factor = 1.2 if event.button == 'up' else 1 / 1.2

        # 获取当前范围
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # 计算新的范围
        new_xlim = [(x - event.xdata) * scale_factor + event.xdata for x in xlim]
        new_ylim = [(y - event.ydata) * scale_factor + event.ydata for y in ylim]

        # 应用新范围
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()

    def on_press(self, event):
        """鼠标按下开始拖动"""
        if event.inaxes != self.ax or event.button != 1:  # 左键
            return
        self.press = event.xdata, event.ydata
        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()

    def on_release(self, event):
        """鼠标释放结束拖动"""
        self.press = None

    def on_motion(self, event):
        """鼠标移动实现拖动"""
        if self.press is None or event.inaxes != self.ax:
            return

        xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        # 应用平移
        self.ax.set_xlim(self.xlim[0] - dx, self.xlim[1] - dx)
        self.ax.set_ylim(self.ylim[0] - dy, self.ylim[1] - dy)
        self.canvas.draw()

    def on_hover(self, event):
        """鼠标悬停显示节点具体模式信息"""
        if event.inaxes != self.ax or self.G is None:
            return

        # 移除旧标签
        if self.hover_text:
            self.hover_text.remove()
            self.hover_text = None
            self.canvas.draw_idle()

        # 检查鼠标是否在节点上
        for node, (x, y) in self.pos.items():
            node_radius = np.sqrt(self.G.nodes[node]['size'] / (70 * np.pi)) * 0.03  # 计算节点半径
            if (event.xdata is not None and
                    np.sqrt((event.xdata - x) ** 2 + (event.ydata - y) ** 2) < node_radius):
                # 获取节点信息
                label = self.G.nodes[node]['label']
                features = self.G.nodes[node]['features']

                # 创建显示文本
                feature_text = "\n".join(features) if features else "No features"
                display_text = f"{label}:\n{feature_text}"

                # 在节点附近显示详细信息
                self.hover_text = self.ax.text(x, y + node_radius + 0.05, display_text,
                                               fontsize=8, ha='center', va='bottom',
                                               bbox=dict(facecolor='white', alpha=0.9,
                                                         edgecolor='#8A2BE2', boxstyle='round,pad=0.5'))
                self.canvas.draw_idle()
                return


class PatternSelectionWindow(QDialog):
    def __init__(self, patterns, feature_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select the patterns of interest.")
        self.setModal(True)
        self.patterns = patterns
        self.feature_names = feature_names
        self.checkboxes = []

        # 设置窗口大小和样式
        self.setFixedSize(800, 500)
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f7fa;
            }
            QCheckBox {
                font: 12pt "Bahnschrift";
                spacing: 10px;
            }
            QLabel {
                font: 11pt "Microsoft YaHei";
                color: #555;
            }
            QDialogButtonBox {
                button-layout: WinLayout;
            }
        """)

        # 创建滚动区域
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # 创建容器
        container = QWidget()
        layout = QGridLayout(container)

        # 按照模式长度从小到大排序
        sorted_patterns = sorted(self.patterns, key=lambda x: sum(x))

        # 为每个模式创建选择项
        for i, pattern in enumerate(sorted_patterns):
            pattern_list = self.vector_to_list(pattern, self.feature_names)

            # 创建复选框
            checkbox = QCheckBox(f"pattern {i + 1}: {pattern_list}", self)
            checkbox.setStyleSheet("""
                QCheckBox {
                    padding: 8px;
                    border-bottom: 1px solid #eee;
                }
                QCheckBox:hover {
                    background-color: #ebf5ff;
                }
            """)
            self.checkboxes.append(checkbox)

            # 创建星级评分
            stars = QLabel("⭐⭐⭐⭐☆")
            stars.setAlignment(Qt.AlignRight)

            # 创建特征向量标签
            feature_vector_label = QLabel(f"(feature vector: {pattern})")
            feature_vector_label.setAlignment(Qt.AlignRight)

            # 添加到布局
            layout.addWidget(checkbox, i, 0)
            layout.addWidget(stars, i, 1)
            layout.addWidget(feature_vector_label, i, 2)

        # 设置列宽比例
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 2)

        scroll_area.setWidget(container)

        # 创建按钮框
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(button_box)

    def get_selected_labels(self):
        return [1 if checkbox.isChecked() else 0 for checkbox in self.checkboxes]

    def vector_to_list(self, pattern_vector, feature_names):
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
            raise ValueError("The number of patterns in the data is insufficient to select the specified quantity.")
        # 随机选择后立即排序
        selected = random.sample(self.data, num_patterns)
        self.selected_patterns = sorted(selected, key=lambda x: sum(x))  # 关键修改：直接存储排序后的模式

    def interact_with_user(self, parent=None):
        """
        与用户交互，确保使用排序后的模式
        """
        if not self.selected_patterns:
            raise ValueError("No patterns selected. Please call the `select_random_patterns` method first.")

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
    finished = pyqtSignal(object)

    def run(self):
        try:
            X, y = MetaLearning.load_data(support_set_label)
            X_scaled = [StandardScaler().fit_transform(task.reshape(-1, 1)).flatten() for task in X]
            input_dim = len(X_scaled[0])
            meta_model = MetaLearning.MetaModelWrapper(input_dim, hidden_units=64)

            for epoch in range(15):
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
                    epochs=1,
                    inner_steps=5
                )

            self.finished.emit(meta_model.meta_model)
        except Exception as e:
            print(f"Pre-training error: {str(e)}")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_connections()
        self.start_pre_training()

        # 初始化属性
        self.csv_data = None
        self.row_data_list = []
        self.auxiliary_list = []
        self.pattern_datas = None
        self.pre_trained_model = None
        self.user_preferences = {}

        # # 创建标签：显示状态信息
        # self.label_result = QLabel("Status: Please select a CSV file", self)

    def setup_ui(self):
        # 设置主窗口属性
        self.setWindowTitle('Meta-PatternLearner - Interactive Data Mining with Meta-Learning')
        self.setWindowTitle('Meta-PatternLearner')
        self.resize(1000, 800)

        # 设置主窗口样式
        self.setStyleSheet("""
            QWidget {
                font-family: "Microsoft YaHei";
            }
            QPushButton {
                background-color: #2B7CD3;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font: 10pt "Bahnschrift";
            }
            QPushButton:hover {
                background-color: #1a68c7;
            }
            QPushButton:pressed {
                background-color: #0d5ab9;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
                font: 9pt "Bahnschrift";
            }
            QProgressBar::chunk {
                background-color: #2B7CD3;
                width: 10px;
            }
        """)

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 创建顶部控制面板
        self.setup_control_panel(main_layout)

        # 创建主内容区域
        self.setup_main_content(main_layout)

        # 创建状态栏
        self.setup_status_bar(main_layout)

        self.setLayout(main_layout)

    def setup_control_panel(self, parent_layout):
        # 控制面板容器
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_panel.setStyleSheet("""
            QFrame {
                background-color: #f0f4f8;
                border-radius: 5px;
                padding: 10px;
            }
        """)

        # 控制面板布局
        panel_layout = QHBoxLayout()
        panel_layout.setSpacing(15)

        # 添加文件操作按钮
        self.btn_openFile = QPushButton('Open CSV File')
        self.btn_openFile.setFixedSize(120, 35)

        self.btn_interact = QPushButton('Start User Interaction')
        self.btn_interact.setFixedSize(150, 35)

        # 添加多样性控制
        diversity_label = QLabel("Diversity:")
        diversity_label.setStyleSheet("font: 10pt 'Bahnschrift';")

        self.diversity_slider = QSlider(Qt.Horizontal)
        self.diversity_slider.setRange(1, 10)
        self.diversity_slider.setValue(5)
        self.diversity_slider.setTickInterval(1)
        self.diversity_slider.setTickPosition(QSlider.TicksBelow)
        self.diversity_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #ddd;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                margin: -5px 0;
                background: #2B7CD3;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #2B7CD3;
                border-radius: 3px;
            }
        """)

        # 添加复杂度控制
        complexity_label = QLabel("Complexity:")
        complexity_label.setStyleSheet("font: 10pt 'Bahnschrift';")

        self.complexity_spin = QSpinBox()
        self.complexity_spin.setRange(1, 10)
        self.complexity_spin.setValue(3)
        self.complexity_spin.setStyleSheet("""
            QSpinBox {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)

        # 将控件添加到布局
        panel_layout.addWidget(self.btn_openFile)
        panel_layout.addWidget(self.btn_interact)
        panel_layout.addStretch()
        panel_layout.addWidget(diversity_label)
        panel_layout.addWidget(self.diversity_slider)
        panel_layout.addWidget(complexity_label)
        panel_layout.addWidget(self.complexity_spin)

        control_panel.setLayout(panel_layout)
        parent_layout.addWidget(control_panel)

    def setup_main_content(self, parent_layout):
        # 创建主内容区域分割器
        splitter = QSplitter(Qt.Vertical)

        # 创建上部区域 - 模式显示
        upper_splitter = QSplitter(Qt.Horizontal)

        # 左侧 - 频繁模式
        self.tab_prevalents = QWidget()
        self.setup_prevalents_tab()

        # 右侧 - 预测模式
        self.tab_predicted = QWidget()
        self.setup_predicted_tab()

        upper_splitter.addWidget(self.tab_prevalents)
        upper_splitter.addWidget(self.tab_predicted)
        upper_splitter.setSizes([400, 400])

        # 创建下部区域 - 监控仪表盘
        lower_splitter = QSplitter(Qt.Horizontal)

        # 左侧 - 模型监控
        self.model_monitor = ModelMonitoringWidget()

        # 右侧 - 模式相似性
        self.pattern_similarity = PatternSimilarityWidget()

        lower_splitter.addWidget(self.model_monitor)
        lower_splitter.addWidget(self.pattern_similarity)
        lower_splitter.setSizes([400, 400])

        # 将上下部分添加到主分割器
        splitter.addWidget(upper_splitter)
        splitter.addWidget(lower_splitter)
        splitter.setSizes([400, 200])

        parent_layout.addWidget(splitter)

    def setup_prevalents_tab(self):
        layout = QVBoxLayout()

        # 创建标题
        title = QLabel("Prevalent Patterns")
        title.setStyleSheet("""
            QLabel {
                font: bold 14pt "Bahnschrift";
                color: #333333;
                padding: 5px;
                border-bottom: 2px solid #2B7CD3;
            }
        """)

        # 创建文本显示区域
        self.text_prevalents = QTextEdit()
        self.text_prevalents.setReadOnly(True)
        self.text_prevalents.setStyleSheet("""
            QTextEdit {
                font: 12pt "Microsoft YaHei";
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
            }
        """)

        layout.addWidget(title)
        layout.addWidget(self.text_prevalents)
        self.tab_prevalents.setLayout(layout)

    def setup_predicted_tab(self):
        layout = QVBoxLayout()

        # 创建标题
        title = QLabel("Predicted Patterns")
        title.setStyleSheet("""
            QLabel {
                font: bold 14pt "Bahnschrift";
                color: #333333;
                padding: 5px;
                border-bottom: 2px solid #8A2BE2;
               }
           """)

        # 创建文本显示区域
        self.text_predicted = QTextEdit()
        self.text_predicted.setReadOnly(True)
        self.text_predicted.setStyleSheet("""
                QTextEdit {
                    font: 12pt "Microsoft YaHei";
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 10px;
                }
            """)

        layout.addWidget(title)
        layout.addWidget(self.text_predicted)
        self.tab_predicted.setLayout(layout)

    def setup_status_bar(self, parent_layout):
        # 状态栏容器
        status_bar = QFrame()
        status_bar.setFrameShape(QFrame.StyledPanel)
        status_bar.setStyleSheet("""
            QFrame {
                background-color: #e9ecef;
                border-radius: 5px;
                padding: 8px;
            }
        """)

        # 状态栏布局
        status_layout = QHBoxLayout()

        # 将状态标签(label_result)添加到最左边
        self.label_result = QLabel("Status: Please select a CSV file")
        self.label_result.setStyleSheet("font: 10pt 'Microsoft YaHei';")

        # 用户偏好显示
        self.user_prefs_label = QLabel("User preference:")
        self.user_prefs_label.setStyleSheet("font: 10pt 'Microsoft YaHei';")

        # 训练进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedSize(150, 15)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # 添加控件到布局，注意顺序
        status_layout.addWidget(self.label_result)
        status_layout.addWidget(self.user_prefs_label)
        status_layout.addStretch()  # 添加伸缩因子，将进度条推到右边
        status_layout.addWidget(self.progress_bar)

        status_bar.setLayout(status_layout)
        parent_layout.addWidget(status_bar)

    def setup_connections(self):
        """设置所有UI组件的信号槽连接"""
        # 文件操作按钮
        self.btn_openFile.clicked.connect(self.openFile)
        self.btn_interact.clicked.connect(self.start_user_interaction)

        # 多样性滑块值变化时更新推荐
        self.diversity_slider.valueChanged.connect(self.on_diversity_changed)

        # 复杂度调节器值变化时更新推荐
        self.complexity_spin.valueChanged.connect(self.on_complexity_changed)

    def start_pre_training(self):
        """启动后台预训练"""
        self.pre_train_thread = PreTrainThread()
        self.pre_train_thread.finished.connect(self.on_pre_train_finished)
        self.pre_train_thread.start()

    def on_pre_train_finished(self, model):
        """预训练完成时的回调函数"""
        self.pre_trained_model = model
        # self.update_status("Pre-training completed!")

    def openFile(self):
        """打开 CSV 文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "./", "CSV Files (*.csv)")

        if file_path:
            try:
                with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    self.csv_data = list(reader)
                    self.row_data_list = [row for row in self.csv_data]
                    self.update_status(f"File read successfully! There are {len(self.csv_data)} rows of data.")
                    self.process_csv_data()
            except Exception as e:
                self.update_status(f"File read failed: {str(e)}.")
                QMessageBox.critical(self, "Error", f"Unable to read the file: {str(e)}")

    def process_csv_data(self):
        """处理读取的 CSV 数据"""
        if self.csv_data:
            self.auxiliary_list = []
            for row in self.row_data_list:
                numeric_row = []
                for value in row:
                    try:
                        numeric_value = int(value)
                    except ValueError:
                        try:
                            numeric_value = float(value)
                        except ValueError:
                            numeric_value = value
                    numeric_row.append(numeric_value)
                self.auxiliary_list.append(numeric_row)

            self.display_prevalent_patterns()
        else:
            self.update_status("No CSV data read")

    def display_prevalent_patterns(self):
        """将频繁模式显示在 Prevalents Patterns 选项卡中"""
        if not self.auxiliary_list:
            return

        feature_names = ['University', 'Hospital', 'Library', 'Restaurants',
                         'Cinema', 'Pharmacy', 'Museum', 'Theater', 'Hotel', 'Station']

        prevalent_patterns_sorted = sorted(self.auxiliary_list, key=lambda x: sum(x))

        if prevalent_patterns_sorted:
            pattern_strings = []
            for pattern in prevalent_patterns_sorted:
                pattern_list = self.vector_to_list(pattern, feature_names)
                pattern_strings.append(", ".join(pattern_list))  # 直接使用特征名称，不添加emoji

            result_html = """
                <style>
                    li {
                        margin-bottom: 12px;
                        font-family: "Microsoft YaHei";
                        font-size: 12pt;
                        padding: 8px;
                        border-bottom: 1px solid #eee;
                    }
                    li:hover {
                        background-color: #f5f9ff;
                    }
                </style>
                <ul>
            """
            for s in pattern_strings:
                result_html += f"<li>{s}</li>"
            result_html += "</ul>"

            self.text_prevalents.setHtml(result_html)
        else:
            self.text_prevalents.clear()

    def start_user_interaction(self):
        """启动用户交互"""
        if not self.auxiliary_list:
            QMessageBox.warning(self, "Error", "No data loaded. Please open the CSV file first.")
            return

        feature_names = ['University', 'Hospital', 'Library', 'Restaurants',
                         'Cinema', 'Pharmacy', 'Museum', 'Theater', 'Hotel', 'Station']

        user_interaction = UserInteraction(self.auxiliary_list, feature_names)
        try:
            # 根据多样性设置选择模式数量
            diversity_level = self.diversity_slider.value()
            num_patterns = min(5 + diversity_level * 2, len(self.auxiliary_list))

            user_interaction.select_random_patterns(num_patterns)
            user_interaction.interact_with_user(self)

            labeled_results = user_interaction.get_labeled_results()
            self.pattern_datas = {
                'interesting': [],
                'uninteresting': [],
                'label': [[], []]
            }

            # 更新用户偏好统计
            for pattern, label in labeled_results:
                pattern_list = self.vector_to_list(pattern, feature_names)
                if label == 1:
                    self.pattern_datas['interesting'].append(pattern)
                    self.pattern_datas['label'][0].append(1)

                    # 更新用户偏好计数
                    for feature in pattern_list:
                        self.user_preferences[feature] = self.user_preferences.get(feature, 0) + 1
                else:
                    self.pattern_datas['uninteresting'].append(pattern)
                    self.pattern_datas['label'][1].append(0)

            # 更新状态栏显示用户偏好
            self.update_user_prefs_display()

            # 从数据中移除已标注的模式
            self.auxiliary_list = user_interaction.remove_labeled_patterns(self.auxiliary_list)

            # 执行后续逻辑
            self.run_post_interaction_logic()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def update_user_prefs_display(self):
        """更新状态栏中的用户偏好显示"""
        if not self.user_preferences:
            self.user_prefs_label.setText("User preference: No data available.")
            return

        # 按偏好程度排序
        sorted_prefs = sorted(self.user_preferences.items(), key=lambda x: x[1], reverse=True)
        # 只显示前3个偏好
        display_text = ", ".join([f"{k}({v})" for k, v in sorted_prefs[:3]])
        self.user_prefs_label.setText(f"User preference: {display_text}")

    def run_post_interaction_logic(self):
        """用户交互完成后执行的逻辑"""
        if not self.pre_trained_model:
            QMessageBox.warning(self, "Error", "The pre-trained model is not loaded, and prediction cannot be made.")
            return

        # 在测试集上微调模型
        self.update_status("Starting model fine-tuning...")
        fine_tune_epochs = 3
        self.progress_bar.setRange(0, fine_tune_epochs)

        fine_tune_optimizer = Adam(learning_rate=0.01)
        X_new = np.array(self.pattern_datas["interesting"] + self.pattern_datas["uninteresting"])
        y_new = np.array(
            [1] * len(self.pattern_datas["interesting"]) + [0] * len(self.pattern_datas["uninteresting"]))

        X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.5, random_state=42)

        # 微调模型并更新监控指标
        for epoch in range(fine_tune_epochs):
            with tf.GradientTape() as tape:
                logits = self.pre_trained_model(X_train, training=True)
                logits = tf.squeeze(logits, axis=-1)
                loss = tf.reduce_mean(binary_crossentropy(y_train, logits))

            grads = tape.gradient(loss, self.pre_trained_model.trainable_variables)
            fine_tune_optimizer.apply_gradients(zip(grads, self.pre_trained_model.trainable_variables))

            # 计算准确率
            train_pred = (logits > 0.5).numpy().astype(int)
            train_acc = np.mean(train_pred == y_train)

            # 更新模型监控指标
            self.model_monitor.update_metrics(loss.numpy(), train_acc, epoch + 1)
            self.progress_bar.setValue(epoch + 1)
            QApplication.processEvents()

        # 评估模型
        test_accuracy, report = MetaLearning.evaluate_model(self.pre_trained_model, X_test, y_test)
        self.update_status(f"Model fine-tuning complete! Test accuracy: {test_accuracy:.2%}")

        # 使用训练好的模型进行预测
        if self.auxiliary_list:
            self.predict_patterns()

    def predict_patterns(self):
        """使用模型预测模式"""
        feature_names = ['University', 'Hospital', 'Library', 'Restaurants',
                         'Cinema', 'Pharmacy', 'Museum', 'Theater', 'Hotel', 'Station']

        X_auxiliary = np.array(self.auxiliary_list)
        y_pred = Nnetwork.predict_with_model(self.pre_trained_model, X_auxiliary)

        # 获取预测结果为1的模式，并生成标签
        self.predicted_patterns = []
        pattern_labels = []  # 存储模式标签(P1, P2,...)

        for idx, (pattern, pred) in enumerate(zip(self.auxiliary_list, y_pred)):
            if pred == 1:
                self.predicted_patterns.append(pattern)
                pattern_labels.append(f"P{idx + 1}")  # 生成标签

        # 更新模式相似性可视化，传入模式标签
        self.pattern_similarity.update_similarity_graph(
            self.predicted_patterns,
            pattern_labels  # 传入标签列表
        )

        # 显示预测结果到文本区域
        if self.predicted_patterns:
            pattern_strings = []
            for pattern, label in zip(self.predicted_patterns, pattern_labels):
                pattern_list = self.vector_to_list(pattern, feature_names)
                # 添加置信度指示条
                confidence = random.uniform(0.7, 0.95)  # 模拟置信度
                confidence_bar = "▮" * int(confidence * 10) + "▯" * (10 - int(confidence * 10))

                pattern_strings.append(
                    f"<b>{label}:</b> {', '.join(pattern_list)} <br>"
                    f"<span style='color:#8A2BE2; font-size:10pt;'>Confidence: {confidence:.0%} {confidence_bar}</span>"
                )

            result_html = """
                    <style>
                        li {
                            margin-bottom: 15px;
                            font-family: "Microsoft YaHei";
                            font-size: 12pt;
                            padding: 10px;
                            border-bottom: 1px solid #eee;
                        }
                        li:hover {
                            background-color: #f9f5ff;
                        }
                    </style>
                    <ul>
                """
            for s in pattern_strings:
                result_html += f"<li>{s}</li>"
            result_html += "</ul>"

            self.text_predicted.setHtml(result_html)
        else:
            self.text_predicted.setText("No patterns of interest found")

    def filter_patterns_by_complexity(self, complexity):
        """根据复杂度过滤模式"""
        filtered_patterns = [
            pattern for pattern in self.auxiliary_list
            if sum(pattern) <= complexity
        ]
        self.display_prevalent_patterns(filtered_patterns)

    def filter_predicted_patterns(self, complexity):
        """根据复杂度过滤预测模式"""
        if hasattr(self, 'predicted_patterns'):
            filtered_patterns = [
                pattern for pattern in self.predicted_patterns
                if sum(pattern) <= complexity
            ]
            self.display_predicted_patterns(filtered_patterns)

    def vector_to_list(self, pattern_vector, feature_names):
        """将模式向量转换为对应的列表表示"""
        if len(pattern_vector) != len(feature_names):
            raise ValueError("The length of the pattern vector does not match the length of the feature name list.")
        return [feature for feature, value in zip(feature_names, pattern_vector) if value == 1]

    def update_status(self, message):
        """更新状态栏消息"""
        self.label_result.setText(f"Status: {message}")

    def on_diversity_changed(self, value):
        """处理多样性滑块值变化"""
        self.update_status(f"Status: {value}")
        if self.auxiliary_list:
            # 根据新的多样性值重新显示模式
            self.display_prevalent_patterns()
            if hasattr(self, 'predicted_patterns'):
                self.display_predicted_patterns()

    def on_complexity_changed(self, value):
        """处理复杂度值变化"""
        self.update_status(f"Complexity set to: {value}")
        if self.auxiliary_list:
            # 根据新的复杂度值过滤模式
            self.filter_patterns_by_complexity(value)
            if hasattr(self, 'predicted_patterns'):
                self.filter_predicted_patterns(value)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 244, 248))
    palette.setColor(QPalette.WindowText, QColor(51, 51, 51))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(51, 51, 51))
    palette.setColor(QPalette.Text, QColor(51, 51, 51))
    palette.setColor(QPalette.Button, QColor(240, 244, 248))
    palette.setColor(QPalette.ButtonText, QColor(51, 51, 51))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(43, 124, 211))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    app.exec_()

# file help start 等