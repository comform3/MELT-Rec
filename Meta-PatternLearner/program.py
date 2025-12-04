import sys
import csv
import random
import itertools
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFileDialog, QMessageBox, QTextEdit, QProgressBar,
    QDialog, QScrollArea, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QTabWidget, QSlider, QSpinBox, QGroupBox, QSplitter, QFrame,
    QListWidget, QDialogButtonBox, QPushButton, QComboBox, QLineEdit, QSizePolicy
)

from PyQt5.QtGui import QColor, QPalette, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from Neuralnetwork import Nnetwork
from PCPs import support_set_label_beijing, support_set_label_shanghai
from Metalearning import MetaLearning
from scipy.interpolate import make_interp_spline


class ModelMonitoringWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.loss_history = []
        self.accuracy_history = []

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # 创建元学习监控组
        group = QGroupBox("Meta-Learning Monitoring")
        group.setStyleSheet("""
            QGroupBox {
                font: bold 14px "Segoe UI";
                border: 2px solid #4A90E2;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4A90E2;
            }
        """)

        # 主布局
        main_layout = QHBoxLayout()

        # 指标面板
        metrics_panel = QFrame()
        metrics_panel.setFrameShape(QFrame.StyledPanel)
        metrics_layout = QVBoxLayout(metrics_panel)

        self.loss_label = QLabel("Loss: ---")
        self.accuracy_label = QLabel("Accuracy: ---")
        self.epoch_label = QLabel("Meta-Epoch: ---")
        self.task_label = QLabel("Current Task: ---")

        for label in [self.loss_label, self.accuracy_label, self.epoch_label, self.task_label]:
            label.setStyleSheet("font: 14px 'Segoe UI'; margin-bottom: 8px;")
            metrics_layout.addWidget(label)

        metrics_layout.addStretch()
        main_layout.addWidget(metrics_panel, 1)

        # 图表面板
        self.figure = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(300, 200)
        self.ax = self.figure.add_subplot(111)

        # 初始化空曲线
        self.loss_curve, = self.ax.plot([], [], 'r-', label='Meta-Loss',
                                        linewidth=2, alpha=0.8)
        self.acc_curve, = self.ax.plot([], [], 'b-', label='Meta-Accuracy',
                                       linewidth=2, alpha=0.8)

        self.ax.set_title('Meta-Learning Progress', fontsize=10)
        self.ax.set_xlabel('Meta-Epoch', fontsize=8)
        self.ax.set_ylabel('Value', fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(fontsize=8)

        main_layout.addWidget(self.canvas, 2)
        group.setLayout(main_layout)
        layout.addWidget(group)
        self.setLayout(layout)

    def _smooth_curve(self, x, y):
        """使用三次样条插值生成平滑曲线"""
        if len(x) < 4:  # 数据点太少时直接返回原数据
            return x, y

        # 生成300个均匀分布的点用于平滑曲线
        x_new = np.linspace(min(x), max(x), 300)

        try:
            # 使用三次样条插值
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_new)
            return x_new, y_smooth
        except:
            return x, y

    def update_metrics(self, loss, accuracy, epoch, task_info=""):
        accuracy = max(0, min(1, accuracy))

        self.loss_label.setText(f"Meta-Loss: {loss:.4f}")
        self.accuracy_label.setText(f"Meta-Accuracy: {accuracy:.4f}")
        self.epoch_label.setText(f"Meta-Epoch: {epoch}")
        self.task_label.setText(f"Current Task: {task_info}")

        # 更新历史数据
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)

        # 准备原始数据
        epochs = np.arange(1, len(self.loss_history) + 1)

        # 生成平滑曲线数据
        x_smooth, loss_smooth = self._smooth_curve(epochs, self.loss_history)
        _, acc_smooth = self._smooth_curve(epochs, self.accuracy_history)

        # 确保平滑后的数据也在0-1范围内（仅对accuracy）
        acc_smooth = np.clip(acc_smooth, 0, 1)

        # 更新曲线
        self.loss_curve.set_data(x_smooth, loss_smooth)
        self.acc_curve.set_data(x_smooth, acc_smooth)

        # 调整坐标轴范围
        self.ax.relim()
        self.ax.autoscale_view()

        # 强制y轴在0-1范围内
        self.ax.set_ylim(0, 1)

        # 确保x轴显示完整
        if len(epochs) > 0:
            self.ax.set_xlim(0.5, max(epochs) + 0.5)

        self.canvas.draw()

    def reset(self):
        """重置元学习监控曲线，用于切换城市或重新开始训练。

        清空历史损失和准确率，并将图表恢复到初始状态。
        """
        # 清空历史数据
        self.loss_history = []
        self.accuracy_history = []

        # 清空曲线数据
        self.loss_curve.set_data([], [])
        self.acc_curve.set_data([], [])

        # 重置坐标轴
        self.ax.clear()
        self.ax.set_title('Meta-Learning Progress', fontsize=10)
        self.ax.set_xlabel('Meta-Epoch', fontsize=8)
        self.ax.set_ylabel('Value', fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.loss_curve, = self.ax.plot([], [], label='Meta-Loss', color='tab:blue')
        self.acc_curve, = self.ax.plot([], [], label='Meta-Accuracy', color='tab:orange')
        self.ax.legend(fontsize=8)

        self.canvas.draw()

    def redraw_from_history(self):
        """根据当前 loss_history / accuracy_history 重绘曲线，用于城市切换后恢复曲线。

        不追加新数据，只是把已有历史重新画一遍，避免影响训练逻辑。
        """
        if not self.loss_history or not self.accuracy_history:
            # 没有历史数据，则清空图像
            self.ax.clear()
            self.ax.set_title('Meta-Learning Progress', fontsize=10)
            self.ax.set_xlabel('Meta-Epoch', fontsize=8)
            self.ax.set_ylabel('Value', fontsize=8)
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(fontsize=8)
            self.canvas.draw()
            return

        epochs = np.arange(1, len(self.loss_history) + 1)
        x_smooth, loss_smooth = self._smooth_curve(epochs, self.loss_history)
        _, acc_smooth = self._smooth_curve(epochs, self.accuracy_history)
        acc_smooth = np.clip(acc_smooth, 0, 1)

        self.loss_curve.set_data(x_smooth, loss_smooth)
        self.acc_curve.set_data(x_smooth, acc_smooth)

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.set_ylim(0, 1)
        if len(epochs) > 0:
            self.ax.set_xlim(0.5, max(epochs) + 0.5)

        self.canvas.draw()

class PatternVisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = None
        self.canvas = None
        self.ax = None
        self.pos = None
        self.G = None
        self.last_patterns = None
        self.setup_ui()
        self.dragged_node = None  # 添加这一行
        self.press_pos = None  # 添加这一行

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
        layout.setContentsMargins(5, 5, 5, 5)

        # 创建景点组合可视化组 - 修改标题和样式
        group = QGroupBox("Attraction Combination Visualization")
        group.setStyleSheet("""
            QGroupBox {
                font: bold 14px "Segoe UI";
                border: 2px solid #8A2BE2;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #8A2BE2;
            }
        """)

        # 创建Matplotlib图表 (保持原样)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(300, 250)
        self.ax = self.figure.add_subplot(111)

        # 初始化空图 (保持原样)
        self.ax.text(0.5, 0.5, 'Waiting for data...',
                     ha='center', va='center',
                     fontsize=12, color='gray')
        self.ax.axis('off')
        self.canvas.draw()

        # 添加控制按钮 (仅修改样式)
        control_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh Visualization")
        self.btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #8A2BE2;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font: 11px "Segoe UI";
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
        group_layout.addWidget(self.btn_refresh)
        group.setLayout(group_layout)


        layout.addWidget(group)
        self.setLayout(layout)

    # 以下是原有功能代码完全保持不变
    def update_similarity_graph(self, patterns, pattern_labels=None):
        """更新相似性图谱
        :param patterns: 景点组合列表
        :param pattern_labels: 与预测列表一致的标签列表（如['C1', 'C2', ...]）
        """
        self.last_patterns = patterns

        if not patterns:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No visible attraction combinations.',
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

            # 计算景点组合之间的相似度
            similarity_matrix = self.calculate_similarity(patterns)

            # 创建图
            self.G = nx.Graph()

            feature_names = ['The Palace Museum', 'Temple of Heaven', 'Summer of Palace', 'Nanluoguxiang',
                             'shichahai', '798 Art District', 'Yonghe Temple', 'National Museum of China', 'Water Cube', 'Mutianyu Great Wall']

            for i, pattern in enumerate(patterns):
                features = [feature_names[j] for j, val in enumerate(pattern) if val == 1]
                label = pattern_labels[i] if pattern_labels else f"C{i + 1}"
                self.G.add_node(i, label=label, size=sum(pattern) * 50 + 100,
                                features=", ".join(features))

            threshold = 0.75
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    if similarity_matrix[i][j] > threshold:
                        self.G.add_edge(i, j, weight=similarity_matrix[i][j] * 5)

            # 计算布局 - 修改这部分参数来优化节点分布
            self.pos = nx.spring_layout(
                self.G,
                k=1.5,  # 增加这个值会使节点间距更大
                iterations=100,  # 增加迭代次数以获得更稳定的布局
                scale=2.0,  # 增加比例因子扩大整体布局
                seed=42,  # 固定随机种子以获得一致的结果
                weight='weight'  # 考虑边的权重
            )

            # 绘制图
            self.ax.clear()

            # 使用蓝色颜色映射
            cmap = plt.cm.Blues

            # 绘制边
            edges = nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax,
                edge_color=[self.G[u][v]['weight'] for u, v in self.G.edges()],
                edge_cmap=cmap,
                width=1.5,
                alpha=0.6
            )

            # 绘制节点
            nodes = nx.draw_networkx_nodes(
                self.G, self.pos, ax=self.ax,
                node_size=[self.G.nodes[n]['size'] for n in self.G.nodes()],
                node_color='#8A2BE2',
                alpha=0.8
            )

            # 在节点内部添加标签
            for node, (x, y) in self.pos.items():
                self.ax.text(x, y, self.G.nodes[node]['label'],
                             fontsize=5, ha='center', va='center',
                             color='white', weight='bold')

            # 设置标题
            self.ax.set_title('Attraction Combination Relationships', fontsize=10)

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
                self.similarity_cbar = self.figure.colorbar(edges, ax=self.ax)
                self.similarity_cbar.set_label('Meta-Similarity')

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

    def show_city_points(self, city_name, feature_names, city_coordinates):
        """在坐标系上展示城市所有景点的固定位置。"""
        if city_name not in city_coordinates:
            return

        coords = city_coordinates.get(city_name, {}) or {}

        self.ax.clear()

        xs, ys, labels = [], [], []
        # 为当前城市定义的所有坐标点画图，不再局限于 feature_names 列表
        for name, (x, y) in coords.items():
            xs.append(x)
            ys.append(y)
            labels.append(name)

        if not xs:
            self.ax.text(0.5, 0.5, 'No coordinates defined for this city.',
                         ha='center', va='center', fontsize=10, color='gray')
            self.ax.axis('off')
            self.canvas.draw()
            return

        self.ax.scatter(xs, ys, s=80, color='#4A90E2', alpha=0.8)
        for x, y, label in zip(xs, ys, labels):
            self.ax.text(x, y + 0.03, label,
                         fontsize=6, ha='center', va='bottom', color='black')

        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(f'{city_name} Attractions Map', fontsize=10)
        self.ax.grid(False)
        self.canvas.draw()

    def calculate_similarity(self, patterns):
        """计算景点组合之间的Jaccard相似度"""
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
        """优化后的悬停显示：仅在悬停节点时显示文本框"""
        # 如果不在图表区域内或没有图数据，移除现有文本框
        if event.inaxes != self.ax or self.G is None or self.pos is None:
            if self.hover_text is not None:
                self.hover_text.remove()
                self.hover_text = None
                self.canvas.draw_idle()
            return

        # 检查鼠标是否悬停在任何节点上
        hovered_node = None
        for node, (x, y) in self.pos.items():
            node_radius = np.sqrt(self.G.nodes[node]['size'] / (70 * np.pi)) * 0.03
            if (event.xdata is not None and
                    np.sqrt((event.xdata - x) ** 2 + (event.ydata - y) ** 2) < node_radius):
                hovered_node = node
                break

        # 如果当前有文本框但鼠标不在节点上，移除文本框
        if self.hover_text is not None and hovered_node is None:
            self.hover_text.remove()
            self.hover_text = None
            self.canvas.draw_idle()
            return

        # 如果鼠标在节点上，显示或更新文本框
        if hovered_node is not None:
            # 获取节点信息
            label = self.G.nodes[hovered_node]['label']
            features = self.G.nodes[hovered_node]['features']
            x, y = self.pos[hovered_node]

            # 如果文本框不存在，创建新的
            if self.hover_text is None:
                self.hover_text = self.ax.annotate(
                    f"{label}: {features}",
                    xy=(x, y),
                    xytext=(0, 20),
                    textcoords='offset pixels',
                    fontsize=7,
                    ha='center',
                    va='bottom',
                    bbox=dict(
                        facecolor='white',
                        alpha=0.9,
                        edgecolor='#8A2BE2',
                        boxstyle='round,pad=1',
                    ),
                    annotation_clip=False,
                )
                self.canvas.draw_idle()
            # 如果文本框已存在，更新位置和内容
            else:
                self.hover_text.xy = (x, y)
                self.hover_text.set_text(f"{label}: {features}")
                self.hover_text.set_position((0, 20))
                self.canvas.draw_idle()

class MetaLearningThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)  # 元学习过程中的错误信息

    def __init__(self, pattern_data, parent=None):
        super().__init__(parent)
        # 保存传进来的支持集（北京或上海）
        self.pattern_data = pattern_data
        # 可选：记录此线程对应的城市名称，由外部在创建后赋值
        self.city_name = None

    def run(self):
        try:
            # 使用传入的 pattern_data，而不是旧的 support_set_label
            X, y = MetaLearning.load_data(self.pattern_data)
            X_scaled = [StandardScaler().fit_transform(task.reshape(-1, 1)).flatten() for task in X]
            input_dim = len(X_scaled[0])
            meta_model = MetaLearning.MetaModelWrapper(input_dim, hidden_units=64)

            # 这里仅执行元学习训练，maml_train 内部负责打印 loss/acc
            # 使用原先的 15 轮预训练设置
            for epoch in range(15):
                # 若外部请求中断，则尽快退出线程
                if self.isInterruptionRequested():
                    print("[Meta-learning] Training thread interrupted, exiting early.")
                    return
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
            # 将错误信息通过信号抛给主界面，在聊天中显示
            self.error.emit(str(e))
            print(f"Pre-training error: {str(e)}")

class MiniMapWidget(FigureCanvas):
    """简易小地图：根据城市景点坐标画一张小图，用于嵌入聊天消息。"""

    def __init__(self, city_name, feature_names, city_coordinates, parent=None):
        # 调大图像尺寸，使在聊天界面中更直观
        fig = Figure(figsize=(6, 4))
        super().__init__(fig)
        self.setParent(parent)
        # 设置更大的最小尺寸，让地图在聊天区域中更宽
        self.setMinimumSize(700, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.ax = fig.add_subplot(111)
        # 统一坐标范围与刻度设置：整个地图为 [0,1]×[0,1]
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # 若为北京或上海，则优先使用对应城市的背景图；否则使用淡色网格背景
        if city_name in ("Beijing", "Shanghai"):
            try:
                # 背景图实际位于 Meta-Learning/assets 目录下
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                filename = "beijing_bg.png" if city_name == "Beijing" else "shanghai_bg.png"
                img_path = os.path.join(base_dir, "assets", filename)
                img = plt.imread(img_path)
                # 以 [0,1]×[0,1] 作为坐标范围铺满背景
                self.ax.imshow(img, extent=(0.0, 1.0, 0.0, 1.0), zorder=0, aspect="auto")
            except Exception as e:
                # 读取失败时，在图中直接显示错误信息，方便排查路径/格式问题
                self.ax.set_facecolor("#EEF2FF")
                msg = f"Failed to load {filename}: {type(e).__name__}"
                print(msg)
                self.ax.text(0.5, 0.5, msg,
                              ha="center", va="center",
                              fontsize=10, color="red",
                              transform=self.ax.transAxes)
        else:
            # 其他城市：保持原来的简洁网格背景
            self.ax.set_facecolor("#EEF2FF")  # 很淡的蓝紫色背景
            for x in np.linspace(0.1, 0.9, 5):
                self.ax.axvline(x, color="#CBD5F5", linewidth=0.6, zorder=0)
            for y in np.linspace(0.1, 0.9, 4):
                self.ax.axhline(y, color="#E0E7FF", linewidth=0.6, zorder=0)

        # 保存点数据用于悬停和拖动
        self.points = []          # 当前点位 (x, y, name)
        self.initial_points = []  # 初始点位副本，用于恢复
        coords = city_coordinates.get(city_name, {}) if city_coordinates else {}
        xs, ys, labels = [], [], []

        # 对当前城市在 city_coordinates 中定义的所有景点画点，
        # 这样即使不在 feature_names 中的额外景点也能出现在地图上
        for name, (x, y) in coords.items():
            xs.append(x)
            ys.append(y)
            labels.append(name)
            self.points.append((x, y, name))

        self.initial_points = list(self.points)

        if not xs:
            self.ax.text(0.5, 0.5, 'No coordinates defined.',
                          ha='center', va='center', fontsize=8, color='gray')
            self.ax.axis('off')
        else:
            # 使用适中的点大小
            self.scatter = self.ax.scatter(xs, ys, s=60, color='#4A90E2', alpha=0.9)
            self.label_texts = []
            for x, y, label in zip(xs, ys, labels):
                t = self.ax.text(x, y + 0.03, label,
                                 fontsize=6, ha='center', va='bottom', color='black')
                self.label_texts.append(t)

            self.ax.set_xlim(0.0, 1.0)
            self.ax.set_ylim(0.0, 1.0)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title(f'{city_name} Attractions Map', fontsize=9)
            self.ax.grid(False)

        fig.tight_layout()

        # 悬停提示框
        self.hover_annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#4A90E2", lw=0.8),
            fontsize=7,
        )
        self.hover_annot.set_visible(False)

        # 交互事件：悬停 + 拖动画布（平移）
        self._press_pos = None
        self._xlim = None
        self._ylim = None
        self.mpl_connect("motion_notify_event", self.on_hover)
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("button_release_event", self.on_release)
        self.mpl_connect("motion_notify_event", self.on_drag)
        self.draw()

    def on_hover(self, event):
        """鼠标悬停在点附近时显示坐标信息。"""
        if event.inaxes != self.ax or not self.points:
            if self.hover_annot.get_visible():
                self.hover_annot.set_visible(False)
                self.draw_idle()
            return

        # 找到最近的点（简单线性搜索，点数不多）
        hovered = None
        min_dist = 0.03  # 判定为“靠近”的半径（在 0-1 坐标系下）
        for x, y, name in self.points:
            if event.xdata is None or event.ydata is None:
                continue
            d = ((event.xdata - x) ** 2 + (event.ydata - y) ** 2) ** 0.5
            if d < min_dist:
                hovered = (x, y, name)
                min_dist = d

        if hovered is None:
            if self.hover_annot.get_visible():
                self.hover_annot.set_visible(False)
                self.draw_idle()
            return

        x, y, name = hovered
        self.hover_annot.xy = (x, y)
        self.hover_annot.set_text(f"{name}\n({x:.2f}, {y:.2f})")
        self.hover_annot.set_visible(True)
        self.draw_idle()

    def on_press(self, event):
        """鼠标按下时记录当前位置，用于后续平移整个坐标系。"""
        if event.inaxes != self.ax or event.button != 1:
            return

        if event.xdata is None or event.ydata is None:
            return

        self._press_pos = (event.xdata, event.ydata)
        self._xlim = self.ax.get_xlim()
        self._ylim = self.ax.get_ylim()

    def on_drag(self, event):
        """拖动过程中平移整个地图视图。"""
        if self._press_pos is None or event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        x0, y0 = self._press_pos
        dx = event.xdata - x0
        dy = event.ydata - y0

        if self._xlim is None or self._ylim is None:
            return

        # 简单平移视图，不做额外限制
        self.ax.set_xlim(self._xlim[0] - dx, self._xlim[1] - dx)
        self.ax.set_ylim(self._ylim[0] - dy, self._ylim[1] - dy)
        self.draw_idle()

    def on_release(self, event):
        """结束拖动。"""
        self._press_pos = None

    def reset_positions(self):
        """恢复到初始坐标布局和视图范围。"""
        if not self.initial_points:
            return

        self.points = list(self.initial_points)

        if hasattr(self, 'scatter'):
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.scatter.set_offsets(list(zip(xs, ys)))

        if hasattr(self, 'label_texts'):
            for t, (px, py, pname) in zip(self.label_texts, self.points):
                t.set_position((px, py + 0.03))

        # 复位视图范围到整张地图 [0,1]×[0,1]
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)

        self.draw_idle()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MELT-Rec")
        self.setWindowIcon(QIcon("icon.png"))  # 请准备一个图标文件
        self.resize(1400, 900)  # 增大窗口尺寸以便更好地显示推荐结果

               # 初始化数据
        self.csv_data = None
        self.auxiliary_list = []
        self.pattern_datas = None
        self.pre_trained_model = None
        self.user_preferences = {}
        self.last_predicted_patterns = None

        # 当前城市，默认北京
        self.current_city = "Beijing"

        # 为每个城市定义各自的景点特征名（10 个特征）
        self.city_feature_names = {
            "Beijing": [
                'The Palace Museum', 'Temple of Heaven', 'Summer of Palace', 'Nanluoguxiang',
                'shichahai', '798 Art District', 'Yonghe Temple', 'National Museum of China',
                'Water Cube', 'Mutianyu Great Wall'
            ],
            "Shanghai": [
                'The Bund', 'Shanghai Disney Resort', 'Lujiazui Financial Center', 'City God Temple',
                'Xintiandi', 'China Art Museum', 'Tianzifang', 'zhujiajiao Ancient Town',
                'Shanghai Museum', 'Qiantan Taikoo Li'
            ],
        }

        # 为每个城市定义写死的景点坐标（归一化到 [0, 1] 区间的示意图坐标）
        self.city_coordinates = {
            "Beijing": {
                # 将景点在 [0,1]×[0,1] 上更均匀铺开，提升视觉丰富度（北京核心 10 个景点恢复为之前的坐标）
                'The Palace Museum': (0.25, 0.80),          # 城中心偏上
                'Temple of Heaven': (0.30, 0.35),           # 城市偏下
                'Summer of Palace': (0.10, 0.70),           # 西北
                'Nanluoguxiang': (0.40, 0.65),              # 中偏上
                'shichahai': (0.35, 0.60),                  # 中偏上
                '798 Art District': (0.80, 0.55),           # 东北
                'Yonghe Temple': (0.45, 0.75),              # 中偏上
                'National Museum of China': (0.55, 0.50),   # 中央略偏右
                'Water Cube': (0.75, 0.40),                 # 东偏下
                'Mutianyu Great Wall': (0.90, 0.90),        # 远东北

                # 额外添加的 6 个北京景点，仅用于地图展示
                'Jingshan Park': (0.30, 0.90),              # 景山公园，故宫正北，略向右
                "Prince Gong's Mansion": (0.18, 0.72),     # 恭王府，偏西北
                'The Ancient Observatory': (0.64, 0.48),    # 古观象台，略向左调整
                'Fayuan Temple': (0.32, 0.25),              # 法源寺，偏南部
                'Coal Hill Park': (0.14, 0.86),             # 煤山公园，偏西北
                'The Capital Museum': (0.08, 0.55),         # 首都博物馆，偏西中部
            },
            "Shanghai": {
                # 将 10 个景点在整张图上左右/上下分散：
                # - 外滩、朱家角偏左；
                # - 陆家嘴、迪士尼偏右；
                # - 其余在中部和中右区域均匀铺开。
                'The Bund': (0.24, 0.86),                   # 北部偏左滨江
                'zhujiajiao Ancient Town': (0.10, 0.24),   # 更靠西南

                'City God Temple': (0.34, 0.38),           # 老城厢中部略偏左，进一步向下调整
                'Xintiandi': (0.40, 0.40),                 # 市中心偏南
                'Tianzifang': (0.50, 0.58),                # 市中心偏上
                'Shanghai Museum': (0.58, 0.76),           # 人民广场一带，略向右调整

                'China Art Museum': (0.60, 0.54),          # 中偏右
                'Qiantan Taikoo Li': (0.70, 0.36),         # 江边新区

                'Lujiazui Financial Center': (0.78, 0.78), # 浦东陆家嘴
                'Shanghai Disney Resort': (0.80, 0.30),    # 浦东东南迪士尼，略向右偏移

                # 额外添加的 6 个上海景点，仅用于地图展示
                'Shanghai Tower': (0.87, 0.86),                 # 上海中心大厦，略向右上调整
                "Jing'an Temple": (0.32, 0.70),                # 静安寺，市中心偏西北
                'Long Museum (West Bund)': (0.28, 0.46),        # 龙美术馆（西岸），黄浦江西岸
                'M50 Creative Park': (0.40, 0.60),              # M50 创意园，略向右下调整
                'Shanghai Film Park': (0.08, 0.40),             # 上海影视乐园，西南郊区偏上
                'Shanghai Natural History Museum': (0.48, 0.80),# 自然博物馆，静安雕塑公园附近
            },
        }

        # 当前使用的特征名
        self.feature_names = self.city_feature_names[self.current_city]
        # 为每个城市维护独立状态（偏好、推荐、模型等）
        self.city_state = {
            "Beijing": {
                "user_preferences": {},
                "last_predicted_patterns": None,
                "all_recommended_patterns": [],
                "recommendation_round": 0,
                "recommendation_rounds": [],
                "selected_interesting": [],
                "selected_uninteresting": [],
                "pre_trained_model": None,
                "meta_learning_completed": False,
                "meta_loss_history": [],
                "meta_accuracy_history": [],
            },
            "Shanghai": {
                "user_preferences": {},
                "last_predicted_patterns": None,
                "all_recommended_patterns": [],
                "recommendation_round": 0,
                "recommendation_rounds": [],
                "selected_interesting": [],
                "selected_uninteresting": [],
                "pre_trained_model": None,
                "meta_learning_completed": False,
                "meta_loss_history": [],
                "meta_accuracy_history": [],
            },
        }

        # 初始化当前城市的状态引用
        state = self.city_state[self.current_city]
        self.user_preferences = state["user_preferences"]
        self.last_predicted_patterns = state["last_predicted_patterns"]
        self.all_recommended_patterns = state["all_recommended_patterns"]
        self.recommendation_round = state["recommendation_round"]
        self.recommendation_rounds = state["recommendation_rounds"]
        self.pre_trained_model = state["pre_trained_model"]
        self.meta_learning_completed = state["meta_learning_completed"]
        self.selected_interesting = state["selected_interesting"]
        self.selected_uninteresting = state["selected_uninteresting"]
        # 聊天交互相关状态
        self.chat_stage = "await_city"  # await_city / await_feedback
        self.chat_current_patterns = []
        self.chat_combo_checkboxes = []
        self.diversity_k = 8  # 每轮候选组合数量（可调）
        self.meta_loading_frame = None  # 元学习预训练时的加载提示

        # 仅设置样式并构建全新的聊天界面壳子
        self.setup_styles()
        self.setup_chat_shell()

        # 不在初始化时直接启动元学习和旧的 UI 流程，后续按需要接入
        self.meta_learning_completed = False

    def setup_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: "Segoe UI";
                font-size: 14px;           /* 全局基础字号再放大一档 */
                background-color: #BFDBFE;  /* 更深一点的天蓝色背景 */
            }
            QPushButton {
                padding: 8px 12px;
                border-radius: 4px;
                font: 14px "Segoe UI";
                min-width: 80px;
            }
            /* 显示框和输入栏使用白色背景，形成层次感 */
            QScrollArea, QScrollArea QWidget {
                background-color: #FFFFFF;
            }
            QLineEdit, QTextEdit, QListWidget {
                border: 1px solid #D1D5DB;
                border-radius: 4px;
                padding: 6px;
                font: 14px "Segoe UI";
                background-color: #FFFFFF;
            }
        """)

    def setup_chat_shell(self):
        """构建一个极简的 ChatGPT 风格界面：
        顶部为消息列表（滚动），底部为输入栏和发送按钮。
        先只支持纯文本消息，后续再接入坐标和组合等复杂消息类型。
        """
        main_layout = QVBoxLayout()
        # 增大整体聊天区域与窗口边缘之间的留白，让蓝色边框更明显（左右收进来一截）
        main_layout.setContentsMargins(40, 20, 40, 20)
        main_layout.setSpacing(8)

        # 系统标题（普通文字）
        title_label = QLabel("MELT-Rec: A Meta-Learning-Based System for Tourism Recommendation")
        title_label.setAlignment(Qt.AlignCenter)

        # 消息列表区域（滚动容器）
        from PyQt5.QtWidgets import QScrollArea

        self.message_area = QScrollArea()
        self.message_area.setWidgetResizable(True)
        # 让中间聊天区域在垂直方向上看起来更大一些
        self.message_area.setMinimumHeight(500)

        container = QWidget()
        self.message_layout = QVBoxLayout(container)
        # 增大左右边距，让所有聊天内容两侧留白更多
        self.message_layout.setContentsMargins(40, 0, 40, 0)
        self.message_layout.setSpacing(6)
        self.message_layout.addStretch()  # 占位，便于后续在末尾插入消息

        self.message_area.setWidget(container)

        # 底部输入栏
        input_layout = QHBoxLayout()
        # 与上方聊天区域之间稍微增加留白，但输入框与按钮之间不留缝隙
        input_layout.setContentsMargins(0, 12, 0, 0)
        input_layout.setSpacing(0)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Enter a city name (Beijing/Shanghai), or type 'continue' to get more recommendations, then press Enter or click Send")
        # 明显增大输入框的高度，使其从上到下更“厚”一些，类似聊天软件的输入栏
        self.input_edit.setMinimumHeight(52)
        # 统一输入框与按钮的字体和边框样式，整体更“轻”，左侧圆角、右侧与按钮无缝拼接
        self.input_edit.setStyleSheet(
            "font: 14px 'Segoe UI';"
            "border: 1px solid #D1D5DB;"           # 更轻的浅灰边框
            "border-right: none;"
            "border-top-left-radius: 4px;"        # 略小的圆角
            "border-bottom-left-radius: 4px;"
            "padding: 6px 10px;"                  # 稍微减小竖向 padding，减轻厚重感
            "background-color: #FFFFFF;"
        )

        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet(
            "background-color: #4A90E2;"          # 保留蓝色作为主视觉
            "color: white;"
            "border: 1px solid #D1D5DB;"         # 与输入框同一浅灰描边
            "border-left: none;"
            "border-top-right-radius: 4px;"
            "border-bottom-right-radius: 4px;"
            "padding: 0 18px;"                   # 略收窄按钮宽度
        )
        self.send_button.setMinimumHeight(52)
        self.send_button.clicked.connect(self.on_send_clicked)
        self.input_edit.returnPressed.connect(self.on_send_clicked)

        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.send_button)

        main_layout.addWidget(title_label)
        main_layout.addWidget(self.message_area, 1)
        main_layout.addLayout(input_layout)

        self.setLayout(main_layout)

        # Initial system hint
        self.add_message(
            "system",
            "Welcome to MELT-Rec.\nEnter a city name to start, then type 'continue' for more recommendations."
        )

    def on_diversity_changed(self, value):
        """当用户调整候选组合数量时更新内部参数。"""
        self.diversity_k = max(1, int(value))

    def add_message(self, role: str, text: str):
        """向消息列表中追加一条消息。role: 'user' 或 'system'

        带简易头像的左右布局，模拟两个好友聊天：
        - user 在右侧，系统在左侧。
        """
        # 最外层 frame 占满整行宽度
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        outer_layout = QHBoxLayout(frame)
        # 默认不留水平内边距，后续在 user 分支中单独微调
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # 中间再包一层固定宽度的列，与推荐景点组合所在列保持一致（720 左右），用于放置头像和气泡
        inner_frame = QFrame()
        inner_frame.setMaximumWidth(720)
        inner_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        inner_layout = QHBoxLayout(inner_frame)
        # 默认使用左右 12px 内边距，用于系统消息列
        inner_layout.setContentsMargins(12, 0, 12, 0)
        inner_layout.setSpacing(8)

        # 文本气泡
        bubble = QLabel(text)
        bubble.setMaximumWidth(1100)
        bubble.setWordWrap(True)
        bubble.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # 简易头像：彩色圆形背景 + 首字母
        avatar = QLabel()
        avatar.setFixedSize(32, 32)
        avatar.setAlignment(Qt.AlignCenter)

        if role.lower() == "user":
            # 用户气泡：右侧，浅绿色背景，统一 14px 字号
            bubble.setStyleSheet(
                "background-color: #DCF8C6; padding: 8px 10px; border-radius: 12px; font: 14px 'Segoe UI';"
            )
            # 自适应但允许更长的单行：将最大宽度放宽到与中间列相同（720）
            bubble.setMaximumWidth(720)

            avatar.setText("U")
            avatar.setStyleSheet(
                "background-color: #2563EB; color: white; border-radius: 16px; font: bold 14px 'Segoe UI';"
            )
            # 为用户行设置右侧 margin≈230px，让整行相对 AlignRight 位置再向右平移约 10px
            outer_layout.setContentsMargins(0, 0, 230, 0)
            # 左侧留空，右侧依次是绿色气泡和头像（恢复上一版布局）
            inner_layout.addStretch()
            inner_layout.addWidget(bubble)
            inner_layout.addWidget(avatar)
        else:
            # 系统气泡：左侧，白底加细边框，统一 14px 字号
            bubble.setStyleSheet(
                "background-color: #FFFFFF; padding: 8px 10px; border-radius: 12px; border: 1px solid #E5E7EB; font: 14px 'Segoe UI';"
            )
            # 固定一个较大的最小宽度，使所有系统气泡都呈现为宽宽的一列，而不是随内容收缩
            bubble.setMinimumWidth(600)
            bubble.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            # 为系统行减少左侧内边距，让头像和白色气泡整体略微向右偏移（左边距从 12 降到 0）
            inner_layout.setContentsMargins(0, 0, 152, 0)
            avatar.setText("M")
            avatar.setStyleSheet(
                "background-color: #10B981; color: white; border-radius: 16px; font: bold 14px 'Segoe UI';"
            )
            # 让系统气泡与推荐景点组合气泡使用同一排版：头像在最左，气泡占据剩余列宽
            inner_layout.addWidget(avatar)
            inner_layout.addWidget(bubble)

        # 头像与消息第一行对齐到顶部
        inner_layout.setAlignment(avatar, Qt.AlignTop)
        inner_layout.setAlignment(bubble, Qt.AlignTop)

        # 让中间这一列与地图所在列位置一致（整体列居中），整行仍然铺满
        outer_layout.addStretch()
        outer_layout.addWidget(inner_frame)
        outer_layout.addStretch()

        # 在倒数第一个 stretch 之前插入，以保持消息从上往下堆叠
        index = self.message_layout.count() - 1
        self.message_layout.insertWidget(index, frame)
        # 根据角色调整整行消息在聊天区域中的水平位置：
        # - user 行靠右
        # - system 行保持默认（靠左）
        if role.lower() == "user":
            self.message_layout.setAlignment(frame, Qt.AlignRight)

        # 自动滚动到底部
        self.message_area.verticalScrollBar().setValue(self.message_area.verticalScrollBar().maximum())

    def on_send_clicked(self):
        text = self.input_edit.text().strip()
        if not text:
            return
        self.add_message("user", text)
        self.input_edit.clear()

        msg = text.lower()

        # 1) 若输入中包含数字，优先尝试按“固定维度 Top-K 推荐”解析
        #    例如输入 "2" 或 "2个景点"，在当前城市的 p>0.5 候选中筛选维度为 2 的组合
        digits = "".join(ch for ch in msg if ch.isdigit())
        if digits:
            try:
                dim_val = int(digits)
            except Exception:
                dim_val = None
            if dim_val is not None and hasattr(self, "feature_names") and 1 <= dim_val <= len(self.feature_names):
                # 由 recommend_topk_for_dimension 自行判断是否已有 p>0.5 的候选
                self.recommend_topk_for_dimension(dim_val, top_k=10)
                return

        # 2) If the user inputs '继续/next/more/continue', start a new recommendation round for the current city
        if msg in {"继续", "next", "more", "continue"}:
            if not getattr(self, "current_city", None):
                self.add_message("system", "No city selected yet. Please enter Beijing or Shanghai first.")
                return
            # 当前城市尚未完成元学习时，提示需要等待
            if self.pre_trained_model is None or not getattr(self, "meta_learning_completed", False):
                self.add_message(
                    "system",
                    f"Meta-learning pre-training is still running for {self.current_city}. Please wait; candidate attraction combinations will be generated automatically once it finishes."
                )
                return

            # 已有预训练模型：基于当前城市重新生成候选景点组合
            try:
                self.generate_all_feature_combinations()
                self.start_chat_combo_selection_round()
            except Exception as e:
                self.add_message("system", f"生成新一轮候选景点组合时出现错误：{e}")
            return

        # 3) 城市识别：北京 / 上海（支持自然语言中包含的城市名，例如 "i want to go to beijing"）
        if "北京" in msg or "beijing" in msg:
            city = "Beijing"
        elif "上海" in msg or "shanghai" in msg:
            city = "Shanghai"
        else:
            self.add_message("system", "Currently only Beijing and Shanghai are supported. Please enter one of these city names.")
            return

        self.current_city = city
        self.feature_names = self.city_feature_names[self.current_city]

        # 确保存在按城市划分的状态字典，并为当前城市初始化状态
        if not hasattr(self, "city_state"):
            self.city_state = {}
        if self.current_city not in self.city_state:
            self.city_state[self.current_city] = {
                "user_preferences": {},
                "last_predicted_patterns": None,
                "all_recommended_patterns": [],
                "recommendation_round": 0,
                "recommendation_rounds": [],
                "selected_interesting": [],
                "selected_uninteresting": [],
                "pre_trained_model": None,
                "meta_learning_completed": False,
            }

        # 进入聊天模式选择城市时，同步当前城市对应的模型和元学习状态
        state = self.city_state[self.current_city]
        self.pre_trained_model = state.get("pre_trained_model", None)
        self.meta_learning_completed = state.get("meta_learning_completed", False)
        self.selected_interesting = state.get("selected_interesting", [])
        self.selected_uninteresting = state.get("selected_uninteresting", [])
        self.add_message(
            "system",
            f"Selected city: {city}. The attractions for this city are shown in the map below."
        )

        # 在聊天中插入一张简易小地图（坐标信息在地图中通过悬停显示）
        mini_map = MiniMapWidget(self.current_city, self.feature_names, self.city_coordinates)
        map_frame = QFrame()
        # 让地图区域与聊天列宽度保持一致，这里与 MiniMapWidget 的可见宽度（700）对齐
        map_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        map_frame.setMaximumWidth(700)

        map_layout = QVBoxLayout(map_frame)
        map_layout.setContentsMargins(12, 4, 12, 4)
        map_layout.setSpacing(4)

        # 地图在局部容器内水平居中显示，整体容器也在消息列中居中
        map_layout.addWidget(mini_map, alignment=Qt.AlignHCenter)

        reset_btn = QPushButton("Reset Positions")
        reset_btn.setStyleSheet("background-color: #E5E7EB; color: #111827; padding: 4px 8px; font: 14px 'Segoe UI';")
        reset_btn.clicked.connect(mini_map.reset_positions)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()

        map_layout.addLayout(btn_row)

        index = self.message_layout.count() - 1
        self.message_layout.insertWidget(index, map_frame)
        # 让地图所在的列在聊天区域中居中
        self.message_layout.setAlignment(map_frame, Qt.AlignHCenter)
        self.message_area.verticalScrollBar().setValue(self.message_area.verticalScrollBar().maximum())

        # 若当前城市尚未完成元学习，则启动预训练；
        # 候选景点组合将在预训练完成后再生成并展示。
        if not getattr(self, "meta_learning_completed", False) or self.pre_trained_model is None:
            # 显示一个可移除的加载提示，而不是长文本消息
            self.show_meta_loading(city)
            try:
                self.start_meta_learning()
            except Exception as e:
                self.hide_meta_loading()
                self.add_message("system", f"启动元学习预训练时出现错误：{e}")
        else:
            # 如果该城市此前已经完成过元学习，则可以直接生成候选组合
            self.generate_all_feature_combinations()
            self.start_chat_combo_selection_round()

    def start_chat_combo_selection_round(self):
        """为当前城市生成一轮景点组合，并以聊天消息形式展示勾选项。"""
        if not self.auxiliary_list:
            self.add_message("system", "There is currently no attraction combination data available.")
            return

        # 简单按多样性选取若干组合（这里暂时固定数量 8，可后续接入滑块参数）
        num_patterns = min(self.diversity_k, len(self.auxiliary_list))
        selected = random.sample(self.auxiliary_list, num_patterns)

        # 传入 self 作为 main_window，让“下一轮候选数量”能够更新 MainWindow.diversity_k
        combo_widget = ChatComboSelectionWidget(selected, self.feature_names, main_window=self)
        combo_widget.feedback_submitted.connect(self.on_chat_combo_feedback)

        # 作为一条 system 消息插入消息列表：候选卡片整体在聊天列中显示
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QHBoxLayout(frame)
        # 将候选列表整体向右偏移约 40 像素：增大左侧 margin
        layout.setContentsMargins(52, 4, 12, 4)

        # 将候选卡片限制在合适宽度范围内，并在 frame 中水平居中
        combo_widget.setMaximumWidth(720)
        layout.addStretch()
        layout.addWidget(combo_widget)
        layout.addStretch()

        index = self.message_layout.count() - 1
        self.message_layout.insertWidget(index, frame)
        self.message_layout.setAlignment(frame, Qt.AlignHCenter)
        self.message_area.verticalScrollBar().setValue(self.message_area.verticalScrollBar().maximum())

    def show_meta_loading(self, city_name: str):
        """在聊天区域显示一个元学习预训练中的加载提示。"""
        # 若已有旧的加载提示，先移除
        self.hide_meta_loading()

        # 内部实际提示框
        inner_frame = QFrame()
        inner_frame.setStyleSheet("background-color: #F3F4F6; border-radius: 8px;")
        inner_layout = QHBoxLayout(inner_frame)
        inner_layout.setContentsMargins(10, 6, 10, 6)
        inner_layout.setSpacing(8)

        dot_label = QLabel("●")
        dot_label.setStyleSheet("color: #4B5563; font: 14px 'Segoe UI';")

        text_label = QLabel(f"Running meta-learning pre-training for {city_name}, please wait…")
        text_label.setStyleSheet("color: #4B5563; font: 14px 'Segoe UI';")
        text_label.setWordWrap(True)

        inner_layout.addWidget(dot_label)
        inner_layout.addWidget(text_label)
        inner_layout.addStretch()

        # 外层包一层，使其作为居中的卡片出现，而不是贴在左侧
        outer_frame = QFrame()
        outer_layout = QHBoxLayout(outer_frame)
        outer_layout.setContentsMargins(12, 4, 12, 4)
        outer_layout.setSpacing(0)

        inner_frame.setMaximumWidth(720)
        outer_layout.addStretch()
        outer_layout.addWidget(inner_frame)
        outer_layout.addStretch()

        index = self.message_layout.count() - 1
        self.message_layout.insertWidget(index, outer_frame)
        self.message_layout.setAlignment(outer_frame, Qt.AlignHCenter)
        self.message_area.verticalScrollBar().setValue(self.message_area.verticalScrollBar().maximum())

        self.meta_loading_frame = outer_frame

    def hide_meta_loading(self):
        """移除元学习预训练的加载提示。"""
        if getattr(self, "meta_loading_frame", None) is not None:
            try:
                self.message_layout.removeWidget(self.meta_loading_frame)
            except Exception:
                pass
            self.meta_loading_frame.hide()
            self.meta_loading_frame.deleteLater()
            self.meta_loading_frame = None

    def on_chat_combo_feedback(self, interesting, uninteresting):
        """处理聊天中提交的景点组合反馈：先总结，再进行一轮推荐并写入聊天。"""
        if not interesting:
            self.add_message("system", "You did not select any attraction combinations as interesting.")
            return
        # If the model is not pre-trained yet, only record preferences and skip recommendations
        if self.pre_trained_model is None or not getattr(self, "meta_learning_completed", False):
            self.add_message(
                "system",
                "Meta-learning pre-training is still running for this city. Your preferences have been recorded; please submit again after pre-training finishes to obtain recommendations."
            )
            # 将本轮反馈缓存到当前城市状态中，便于预训练完成后复用
            self.selected_interesting = interesting
            self.selected_uninteresting = uninteresting
            return

        # 2) 利用反馈微调模型
        try:
            self.fine_tune_model(interesting, uninteresting)
        except Exception as e:
            self.add_message("system", f"微调模型时出现错误：{e}")
            return

        # 3) 基于微调后的模型预测新的景点组合
        try:
            self.predict_new_patterns()
        except Exception as e:
            self.add_message("system", f"生成推荐结果时出现错误：{e}")
            return

        # Get the latest recommendation results from recommendation_rounds
        rounds = getattr(self, "recommendation_rounds", [])
        if not rounds or not rounds[-1]:
            self.add_message("system", "Based on your current preferences, there are no new attraction combinations to recommend.")
            return

        recommended = rounds[-1]
        rec_lines = []
        for idx, pattern in enumerate(recommended, start=1):
            feats = [self.feature_names[j] for j, val in enumerate(pattern) if val == 1]
            rec_lines.append(f"{idx}. {', '.join(feats)}")
        # Also add a blank line between recommended combinations for readability
        rec_text = "\n\n".join(rec_lines)

        self.add_message(
            "system",
            "Based on your feedback, we recommend the following attraction combinations:\n" + rec_text
        )

    def setup_header(self, layout):
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4A90E2, stop:1 #8A2BE2);
                border-radius: 8px;
                padding: 15px;
            }
        """)

        header_layout = QVBoxLayout(header)

        title = QLabel("MELT-Rec")
        title.setStyleSheet("""
            QLabel {
                font: bold 16px "Segoe UI";
                color: white;
                qproperty-alignment: AlignCenter;
            }
        """)

        subtitle = QLabel("Interactive Data Mining with Meta-Learning")
        subtitle.setStyleSheet("""
            QLabel {
                font: 14px "Segoe UI";
                color: rgba(255,255,255,0.9);
                qproperty-alignment: AlignCenter;
                margin-top: 5px;
            }
        """)

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)

        # 城市选择行
        city_row = QHBoxLayout()
        city_row.setContentsMargins(0, 10, 0, 0)
        city_row.setSpacing(10)

        city_label = QLabel("City:")
        city_label.setStyleSheet("""
            QLabel {
                font: 14px "Segoe UI";
                color: rgba(255,255,255,0.95);
            }
        """)

        self.city_combo = QComboBox()
        # 这里的名字要和 self.city_feature_names 的 key 一致
        self.city_combo.addItems(["Beijing", "Shanghai"])
        self.city_combo.setCurrentText(self.current_city)
        self.city_combo.setStyleSheet("""
            QComboBox {
                font: 14px "Segoe UI";
                padding: 4px 8px;
                border-radius: 4px;
                background-color: rgba(255,255,255,0.9);
                color: #333333;
                min-width: 120px;
            }
        """)
        self.city_combo.currentTextChanged.connect(self.on_city_changed)

        city_row.addStretch()
        city_row.addWidget(city_label)
        city_row.addWidget(self.city_combo)
        city_row.addStretch()

        header_layout.addLayout(city_row)
        layout.addWidget(header)

    def setup_main_content(self, layout):
        # 使用一个隐藏 TabWidget 承载 Chat / Data / Interaction，其中 Chat 是唯一可见的主界面
        left_panel = QTabWidget()
        left_panel.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #D1D5DB;
                border-radius: 6px;
            }
            QTabBar::tab {
                padding: 8px 15px;
                font: bold 14px "Segoe UI";
                background: #EDF2F7;
            }
            QTabBar::tab:selected {
                background: #4A90E2;
                color: white;
            }
        """)

        # 先创建聊天式交互选项卡，作为默认主界面
        self.setup_chat_tab(left_panel)

        # 仍然创建 Data / Interaction 页，用于复用其中的控件和逻辑，但隐藏 Tab 栏
        self.setup_data_tab(left_panel)
        self.setup_interaction_tab(left_panel)

        # 隐藏标签栏，使界面看起来就是一个纯 Chat 窗口
        if left_panel.tabBar() is not None:
            left_panel.tabBar().setVisible(False)
        left_panel.setCurrentIndex(0)

        # 元学习监控小部件仍然存在以供内部更新，但不再单独显示一个页面
        self.model_monitor = ModelMonitoringWidget()

        # 将整个 left_panel 作为主内容添加到布局中
        layout.addWidget(left_panel)

    def setup_chat_tab(self, tab_widget):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setMinimumHeight(150)
        self.chat_history.setStyleSheet("font: 14px 'Segoe UI';")

        # 城市景点坐标示意图（固定坐标的小地图）
        self.pattern_viz = PatternVisualizationWidget()
        self.pattern_viz.setMinimumHeight(220)

        # 组合勾选区域
        combo_group = QGroupBox("Current Attraction Combinations")
        combo_layout = QVBoxLayout(combo_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # 根据候选个数设置一个合适的最小高度，使得一轮 8 个候选基本可以完整展示
        visible_rows = min(len(self.patterns), 8)
        row_height = 32  # 每条候选大致占用的高度（含行距）
        scroll.setMinimumHeight(visible_rows * row_height + 20)
        container = QWidget()
        self.chat_combo_container_layout = QVBoxLayout(container)
        scroll.setWidget(container)

        self.chat_submit_btn = QPushButton("Submit Feedback")
        self.chat_submit_btn.setStyleSheet("background-color: #4A90E2; color: white;")
        self.chat_submit_btn.clicked.connect(self.on_chat_submit_feedback)

        combo_layout.addWidget(scroll)
        combo_layout.addWidget(self.chat_submit_btn)

        # 底部输入区
        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Enter a city name, e.g. Beijing or Shanghai, then press Enter or click Send")
        self.chat_send_btn = QPushButton("Send")
        self.chat_send_btn.setStyleSheet("background-color: #8A2BE2; color: white;")
        self.chat_send_btn.clicked.connect(self.on_chat_send)
        self.chat_input.returnPressed.connect(self.on_chat_send)

        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(self.chat_send_btn)

        layout.addWidget(self.chat_history)
        # 让城市地图和候选列表在主界面中水平居中显示
        layout.addWidget(self.pattern_viz)
        layout.setAlignment(self.pattern_viz, Qt.AlignHCenter)
        layout.addWidget(combo_group, 2)
        layout.setAlignment(combo_group, Qt.AlignHCenter)
        layout.addLayout(input_layout)

        tab.setLayout(layout)
        tab_widget.addTab(tab, "Chat")

        # Initial system hint (legacy text area)
        self.append_chat("System", "Please first enter a city name, for example: Beijing or Shanghai.")

    def append_chat(self, role, text):
        if not hasattr(self, "chat_history") or self.chat_history is None:
            return
        prefix = "User" if role.lower() == "user" else "System"
        current = self.chat_history.toPlainText()
        new_line = f"{prefix}: {text}"
        if current:
            self.chat_history.setPlainText(current + "\n" + new_line)
        else:
            self.chat_history.setPlainText(new_line)
        if self.chat_history.verticalScrollBar():
            self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def on_chat_send(self):
        text = self.chat_input.text().strip() if hasattr(self, "chat_input") else ""
        if not text:
            return
        self.append_chat("User", text)
        self.chat_input.clear()
        self.handle_chat_message(text)

    def handle_chat_message(self, text):
        msg = text.strip().lower()
        
        # 只要输入中包含数字，就优先尝试按“固定维度 Top-K 推荐”解析，
        # 让 recommend_topk_for_dimension 自行判断当前是否已有推荐结果。
        digits = "".join(ch for ch in msg if ch.isdigit())
        if digits:
            try:
                dim_val = int(digits)
            except Exception:
                dim_val = None
            if dim_val is not None and 1 <= dim_val <= len(self.feature_names):
                self.recommend_topk_for_dimension(dim_val, top_k=10)
                return

        if self.chat_stage == "await_city":
            city = None
            # 既支持纯城市名（"beijing"），也支持自然语言句子中包含的城市名
            if "北京" in msg or "beijing" in msg:
                city = "Beijing"
            elif "上海" in msg or "shanghai" in msg:
                city = "Shanghai"

            if city is None:
                self.append_chat("System", "Currently only Beijing and Shanghai are supported. Please enter one of these city names.")
                return

            # 切换城市（会触发 on_city_changed）
            self.city_combo.setCurrentText(city)

            # Text reply about the selected city (legacy text area)
            self.append_chat(
                "System",
                f"Selected city: {city}. The attractions for this city are shown in the map above."
            )

            # 更新上方城市坐标示意图
            try:
                if hasattr(self, "pattern_viz"):
                    self.pattern_viz.show_city_points(self.current_city, self.feature_names, self.city_coordinates)
            except Exception:
                pass

            # 生成第一轮推荐并在 Chat 中展示
            self.start_chat_recommendation_round()
            self.chat_stage = "await_feedback"
            return

        if self.chat_stage == "await_feedback":
            # 若无法解析出维度，继续提示用户在中间列表中勾选并提交反馈
            self.append_chat(
                "System",
                "Please select the attraction combinations you are interested in in the middle list, then click the Submit Feedback button."
            )

    def start_chat_recommendation_round(self):
        """为当前城市生成一轮景点组合并在 Chat 选项卡中显示勾选项。"""
        if not self.auxiliary_list:
            QMessageBox.warning(self, "No Data", "No attraction combinations are available.")
            return

        # 根据多样性设置选择景点组合数量（与原始交互逻辑保持一致）
        num_patterns = min(5 + self.diversity_slider.value() * 2, len(self.auxiliary_list))
        selected = random.sample(self.auxiliary_list, num_patterns)
        selected_sorted = sorted(selected, key=lambda x: sum(x))

        # 清空旧的组合条目
        if hasattr(self, "chat_combo_container_layout"):
            while self.chat_combo_container_layout.count():
                item = self.chat_combo_container_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)

        self.chat_current_patterns = selected_sorted
        self.chat_combo_checkboxes = []

        feature_names = self.feature_names

        for i, pattern in enumerate(self.chat_current_patterns):
            group = QGroupBox()
            group.setStyleSheet(
                "QGroupBox { border: 1px solid #E2E8F0; border-radius: 4px; margin-top: 5px; }"
            )
            row = QHBoxLayout(group)

            features = [feature_names[j] for j, val in enumerate(pattern) if val == 1]
            item_text = f"Combination {i + 1}: {', '.join(features)}"

            cb = QCheckBox(item_text)
            cb.setStyleSheet("font: 14px 'Segoe UI';")
            self.chat_combo_checkboxes.append(cb)

            complexity = sum(pattern)
            stars = "★" * complexity + "☆" * (len(feature_names) - complexity)
            complexity_label = QLabel(stars)
            complexity_label.setStyleSheet("color: #8A2BE2; font: 14px 'Segoe UI';")

            row.addWidget(cb)
            row.addWidget(complexity_label)

            self.chat_combo_container_layout.addWidget(group)

        # 旧版 Chat Tab 提示语已由新的单窗口聊天界面替代，这里不再重复输出

    def on_chat_submit_feedback(self):
        """从 Chat 组合列表读取用户反馈，并复用原有微调与推荐逻辑。"""
        if not self.chat_current_patterns or not self.chat_combo_checkboxes:
            QMessageBox.information(self, "No Selection", "There are currently no combinations to submit.")
            return

        interesting = []
        uninteresting = []
        for pattern, cb in zip(self.chat_current_patterns, self.chat_combo_checkboxes):
            if cb.isChecked():
                interesting.append(pattern)
            else:
                uninteresting.append(pattern)

        if not interesting:
            QMessageBox.information(self, "No Selection", "You didn't select any interesting attraction combinations")
            return

        # 在聊天历史中总结用户选择
        feature_names = self.feature_names
        lines = []
        for idx, pattern in enumerate(interesting, start=1):
            feats = [feature_names[j] for j, val in enumerate(pattern) if val == 1]
            lines.append(f"{idx}. {', '.join(feats)}")
        summary = "\n".join(lines)
        self.append_chat("System", f"你选择的感兴趣景点组合有：\n{summary}")

        # 复用原有逻辑：更新已选样本，并进入微调+预测
        self.selected_interesting = interesting
        self.selected_uninteresting = uninteresting

        if self.meta_learning_completed:
            self.fine_tune_and_predict()
        else:
            if hasattr(self, "status_label"):
                self.status_label.setText("Attraction combinations selected. Waiting for meta-learning to complete...")
            self.append_chat(
                "System",
                "Your preferences have been recorded. Meta-learning pre-training is being completed, and further recommendations will be provided shortly."
            )

    def setup_data_tab(self, tab_widget):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 数据信息组
        file_group = QGroupBox("Available Attraction Combinations")
        file_group.setStyleSheet("""
            QGroupBox {
                font: bold 14px "Segoe UI";
                border: 2px solid #4A90E2;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4A90E2;
            }
        """)

        file_layout = QVBoxLayout(file_group)

        self.data_info = QLabel("Generating feature combinations...")
        self.data_info.setStyleSheet("font: 14px 'Segoe UI'; color: #666;")

        file_layout.addWidget(self.data_info)
        file_layout.addStretch()

        # 数据显示
        self.data_display = QListWidget()
        self.data_display.setStyleSheet("font: 13px 'Consolas';")

        layout.addWidget(file_group)
        layout.addWidget(self.data_display)
        tab.setLayout(layout)
        tab_widget.addTab(tab, "Data")

    def setup_interaction_tab(self, tab_widget):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 交互控制组
        control_group = QGroupBox("User Interaction")
        control_group.setStyleSheet("""
            QGroupBox {
                font: bold 14px "Segoe UI";
                border: 2px solid #8A2BE2;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #8A2BE2;
            }
        """)

        control_layout = QVBoxLayout(control_group)

        btn_start = QPushButton("Start Interaction")
        btn_start.setStyleSheet("background-color: #8A2BE2; color: white;")
        btn_start.clicked.connect(self.start_interaction)

        # 参数控制
        param_layout = QHBoxLayout()

        diversity_label = QLabel("Diversity:")
        self.diversity_slider = QSlider(Qt.Horizontal)
        self.diversity_slider.setRange(1, 10)
        self.diversity_slider.setValue(5)

        complexity_label = QLabel("Complexity:")
        self.complexity_spin = QSpinBox()
        self.complexity_spin.setRange(1, 10)
        self.complexity_spin.setValue(3)

        param_layout.addWidget(diversity_label)
        param_layout.addWidget(self.diversity_slider)
        param_layout.addWidget(complexity_label)
        param_layout.addWidget(self.complexity_spin)

        control_layout.addWidget(btn_start)
        control_layout.addLayout(param_layout)
        control_layout.addStretch()

        # 结果显示 - 增大推荐结果显示区域
        result_group = QGroupBox("Recommended Attraction Combinations")
        result_group.setStyleSheet("""
            QGroupBox {
                font: bold 14px "Segoe UI";
                border: 2px solid #4A90E2;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4A90E2;
            }
        """)
        result_layout = QVBoxLayout(result_group)
        
        # Breadth 筛选滑块（与 Diversity 滑块配置一致，保持丝滑体验）
        breadth_filter_layout = QHBoxLayout()
        breadth_filter_label = QLabel("Breadth:")
        self.breadth_slider = QSlider(Qt.Horizontal)
        self.breadth_slider.setRange(2, len(self.feature_names))
        default_breadth = min(3, len(self.feature_names))
        self.breadth_slider.setValue(default_breadth)
        self.breadth_value = QLabel(f"{default_breadth} attractions")
        self.breadth_slider.valueChanged.connect(self.on_breadth_changed)
        breadth_filter_layout.addWidget(breadth_filter_label)
        breadth_filter_layout.addWidget(self.breadth_slider)
        breadth_filter_layout.addWidget(self.breadth_value)
        result_layout.addLayout(breadth_filter_layout)
        
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setMinimumHeight(300)  # 增大最小高度
        self.result_display.setStyleSheet("""
            QTextEdit {
                font: 13px "Segoe UI";
                padding: 10px;
                border: 2px solid #4A90E2;
                border-radius: 6px;
                background-color: white;
            }
        """)
        result_layout.addWidget(self.result_display)

        layout.addWidget(control_group)
        layout.addWidget(result_group, 2)  # 给推荐结果区域更大的权重
        tab.setLayout(layout)
        tab_widget.addTab(tab, "Interaction")

    def setup_status_bar(self, layout):
        status_bar = QFrame()
        status_bar.setStyleSheet("""
            QFrame {
                background-color: #EDF2F7;
                border-radius: 6px;
                padding: 8px;
                border: 1px solid #D1D5DB;
            }
        """)

        status_layout = QHBoxLayout(status_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font: 14px 'Segoe UI';")

        self.prefs_label = QLabel("User preferences: None")
        self.prefs_label.setStyleSheet("font: 14px 'Segoe UI'; color: #8A2BE2;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.prefs_label)
        status_layout.addStretch()
        status_layout.addWidget(self.progress_bar)

        layout.addWidget(status_bar)

    def on_city_changed(self, city_name: str):
        """当用户切换城市时，更新特征、数据和模型"""
        if not hasattr(self, "city_feature_names"):
            return
        if city_name not in self.city_feature_names:
            return
        # 先保存旧城市的状态
        old_city = self.current_city
        if hasattr(self, "city_state") and old_city in self.city_state:
            state = self.city_state[old_city]
            state["user_preferences"] = self.user_preferences.copy() if hasattr(self, "user_preferences") else {}
            state["last_predicted_patterns"] = getattr(self, "last_predicted_patterns", None)
            state["all_recommended_patterns"] = getattr(self, "all_recommended_patterns", []).copy()
            state["recommendation_round"] = getattr(self, "recommendation_round", 0)
            state["recommendation_rounds"] = getattr(self, "recommendation_rounds", []).copy()
            state["selected_interesting"] = getattr(self, "selected_interesting", []).copy()
            state["selected_uninteresting"] = getattr(self, "selected_uninteresting", []).copy()
            state["pre_trained_model"] = getattr(self, "pre_trained_model", None)
            state["meta_learning_completed"] = getattr(self, "meta_learning_completed", False)

        # 更新当前城市
        self.current_city = city_name
        self.feature_names = self.city_feature_names[self.current_city]

        if hasattr(self, "city_state") and self.current_city in self.city_state:
            state = self.city_state[self.current_city]
            self.user_preferences = state.get("user_preferences", {})
            self.last_predicted_patterns = state.get("last_predicted_patterns", None)
            self.all_recommended_patterns = state.get("all_recommended_patterns", [])
            self.recommendation_round = state.get("recommendation_round", 0)
            self.recommendation_rounds = state.get("recommendation_rounds", [])
            self.selected_interesting = state.get("selected_interesting", [])
            self.selected_uninteresting = state.get("selected_uninteresting", [])
            self.pre_trained_model = state.get("pre_trained_model", None)
            self.meta_learning_completed = state.get("meta_learning_completed", False)

        # 重新生成所有特征组合及显示
        self.generate_all_feature_combinations()
        self.display_data_samples()

        # 更新 Breadth 滑块范围和显示
        self.breadth_slider.blockSignals(True)
        self.breadth_slider.setRange(2, len(self.feature_names))
        default_breadth = min(3, len(self.feature_names))
        self.breadth_slider.setValue(default_breadth)
        self.breadth_value.setText(f"{default_breadth} attractions")
        self.breadth_slider.blockSignals(False)

        # 更新偏好显示和结果区域
        self.update_prefs_display()
        self.result_display.clear()
        if self.last_predicted_patterns:
            # 有历史推荐则直接展示
            self.display_recommendations(self.last_predicted_patterns, None)
        else:
            self.result_display.append("Please start interaction to get recommendations.")

        # 尝试停止之前的预训练线程（如果还在运行，仅做防御性尝试）
        if hasattr(self, 'pre_train_thread') and self.pre_train_thread.isRunning():
            try:
                self.pre_train_thread.requestInterruption()
            except Exception:
                pass

        # 根据当前城市是否已有预训练模型决定是否重新跑元学习
        if not getattr(self, "meta_learning_completed", False) or self.pre_trained_model is None:
            if hasattr(self, "status_label"):
                self.status_label.setText(f"Switched to {self.current_city}. Re-running meta-learning...")
            if hasattr(self, "progress_bar"):
                self.progress_bar.setValue(0)
            self.start_meta_learning()
        else:
            if hasattr(self, "status_label"):
                self.status_label.setText(f"Switched to {self.current_city}. Meta-learning already completed.")

    def start_meta_learning(self):
        """启动后台预训练"""
        # 如果已经有一个预训练线程在跑：
        # - 若是当前城市的训练，则不要重复启动；
        # - 若是其他城市的训练（通常已经在 on_city_changed 中 requestInterruption），
        #   允许为新城市启动新的线程，旧线程会尽快因为中断退出。
        if hasattr(self, 'pre_train_thread') and self.pre_train_thread.isRunning():
            running_city = getattr(self.pre_train_thread, 'city_name', None)
            if running_city == self.current_city:
                print("[Meta-learning] Training thread for this city is already running, skip starting a new one.")
                return

        if self.current_city == "Beijing":
            pattern_data = support_set_label_beijing
        else:
            pattern_data = support_set_label_shanghai

        print(f"[Meta-learning] City = {self.current_city}, "
              f"interesting = {len(pattern_data['interesting'])}, "
              f"uninteresting = {len(pattern_data['uninteresting'])}")

        self.pre_train_thread = MetaLearningThread(pattern_data)
        # 标记这个线程是为哪个城市跑的
        self.pre_train_thread.city_name = self.current_city
        self.pre_train_thread.finished.connect(self.on_meta_learning_finished)
        # 将错误信号连接到统一的错误处理函数
        self.pre_train_thread.error.connect(self.on_meta_learning_error)
        self.pre_train_thread.start()

    def update_meta_metrics(self, loss, accuracy, epoch, task_info):
        """更新元学习指标"""
        if hasattr(self, "model_monitor"):
            self.model_monitor.update_metrics(loss, accuracy, epoch, task_info)
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(int((epoch / 15) * 100))

    def on_meta_learning_finished(self, model):
        """元学习完成回调"""
        self.pre_trained_model = model
        self.meta_learning_completed = True

        # 将训练好的模型和状态写回当前城市的 city_state，保证每个城市各自维护一套模型
        if hasattr(self, "city_state") and self.current_city in self.city_state:
            state = self.city_state[self.current_city]
            state["pre_trained_model"] = self.pre_trained_model
            state["meta_learning_completed"] = self.meta_learning_completed
        if hasattr(self, "status_label"):
            self.status_label.setText("Meta-learning completed! Ready for fine-tuning.")
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(100)

        # 移除加载提示，并在聊天中提示当前城市的元学习已完成
        try:
            self.hide_meta_loading()
            self.add_message(
                "system",
                f"Meta-learning pre-training for {self.current_city} has finished. We can now generate a recommendation round based on your selected preferences."
            )
        except Exception:
            pass
        # 如果在预训练期间已经缓存了用户反馈，则自动基于该反馈执行一轮微调+推荐
        if getattr(self, "selected_interesting", None):
            try:
                self.fine_tune_and_predict()
            except Exception as e:
                try:
                    self.add_message("system", f"基于你之前选择进行微调时出现错误：{e}")
                except Exception:
                    pass
                return

            # 将最新一轮推荐结果同步到聊天中
            rounds = getattr(self, "recommendation_rounds", [])
            if rounds and rounds[-1]:
                feature_names = self.feature_names
                recommended = rounds[-1]
                rec_lines = []
                for idx, pattern in enumerate(recommended, start=1):
                    feats = [feature_names[j] for j, val in enumerate(pattern) if val == 1]
                    rec_lines.append(f"{idx}. {', '.join(feats)}")
                rec_text = "\n".join(rec_lines)
                try:
                    self.add_message(
                        "system",
                        "基于你在预训练前提交的偏好，我们为你推荐以下景点组合：\n" + rec_text
                    )
                except Exception:
                    pass
        else:
            # 如果没有缓存的反馈，则在预训练完成后生成第一轮候选景点组合
            try:
                self.generate_all_feature_combinations()
                self.start_chat_combo_selection_round()
            except Exception as e:
                try:
                    self.add_message("system", f"生成候选景点组合时出现错误：{e}")
                except Exception:
                    pass
    def on_meta_learning_error(self, message: str):
        """在聊天界面中显示元学习预训练过程中的错误信息。"""
        # 确保不会因为聊天控件未初始化而再次抛错
        try:
            self.hide_meta_loading()
            self.add_message(
                "system",
                f"An error occurred during meta-learning pre-training: {message}"
            )
        except Exception:
            # 最坏情况：无法写入聊天，只打印到控制台
            print(f"[Meta-learning][UI] Error: {message}")

    def fine_tune_and_predict(self):
        """执行微调和预测"""
        self.fine_tune_model(self.selected_interesting, self.selected_uninteresting)
        self.predict_new_patterns()

    def generate_all_feature_combinations(self):
        """生成所有特征的所有组合"""
        num_features = len(self.feature_names)
        
        # 生成所有可能的组合（从2个特征到所有特征）
        all_combinations = []
        for r in range(2, num_features + 1):  # 从2个特征开始，直到所有特征
            combinations = itertools.combinations(range(num_features), r)
            for combo in combinations:
                all_combinations.append(combo)
        
        # 将组合转换为二进制向量
        self.auxiliary_list = []
        for combo in all_combinations:
            # 创建二进制向量
            binary_pattern = [1 if i in combo else 0 for i in range(num_features)]
            self.auxiliary_list.append(binary_pattern)
        
        # 更新数据信息显示
        if hasattr(self, 'data_info'):
            self.data_info.setText(f"Generated: {len(self.auxiliary_list)} attraction combinations from {num_features} attractions")
        
        print(f"Generated {len(self.auxiliary_list)} feature combinations from {num_features} features")
        print(f"Feature names: {', '.join(self.feature_names)}")

    def display_data_samples(self):
        """显示数据样本"""
        self.data_display.clear()

        # 使用动态特征名（如果有）或默认特征名
        if hasattr(self, 'feature_names') and self.feature_names:
            feature_names = self.feature_names
        else:
            feature_names = ['The Palace Museum', 'Temple of Heaven', 'Summer of Palace', 'Nanluoguxiang',
                             'shichahai', '798 Art District', 'Yonghe Temple', 'National Museum of China', 'Water Cube', 'Mutianyu Great Wall']

        for i, pattern in enumerate(self.auxiliary_list[:50]):  # 显示前50个样本
            if isinstance(pattern, list) and all(isinstance(x, (int, float)) for x in pattern):
                # 二进制向量格式
                features = [feature_names[j] for j, val in enumerate(pattern) 
                           if j < len(feature_names) and val == 1]
                item_text = f"Combination {i + 1}: {', '.join(features) if features else 'Empty'}"
            else:
                # 其他格式
                item_text = f"Combination {i + 1}: {pattern}"
            
            self.data_display.addItem(item_text)

    def start_interaction(self):
        """启动用户交互"""
        if not self.auxiliary_list:
            QMessageBox.warning(self, "No Data", "No attraction combinations available")
            return

        # 根据多样性设置选择景点组合数量
        num_patterns = min(5 + self.diversity_slider.value() * 2, len(self.auxiliary_list))

        # 随机选择景点组合
        selected = random.sample(self.auxiliary_list, num_patterns)
        selected_sorted = sorted(selected, key=lambda x: sum(x))

        # 显示交互对话框
        dialog = PatternSelectionDialog(selected_sorted, self)
        if dialog.exec_() == QDialog.Accepted:
            self.process_user_feedback(dialog.get_selected_patterns())
        else:
            self.status_label.setText("Interaction canceled")

    def process_user_feedback(self, selected_patterns):
        """处理用户反馈"""
        interesting = [p for p, selected in selected_patterns if selected]
        uninteresting = [p for p, selected in selected_patterns if not selected]

        if not interesting:
            QMessageBox.information(self, "No Selection", "You didn't select any interesting attraction combinations")
            return

        # 更新用户偏好
        feature_names = ['The Palace Museum', 'Temple of Heaven', 'Summer of Palace', 'Nanluoguxiang',
                         'shichahai', '798 Art District', 'Yonghe Temple', 'National Museum of China', 'Water Cube', 'Mutianyu Great Wall']

        for pattern in interesting:
            for i, val in enumerate(pattern):
                if val == 1:
                    feature = feature_names[i]
                    self.user_preferences[feature] = self.user_preferences.get(feature, 0) + 1

        # 更新偏好显示
        self.update_prefs_display()

        # 存储选择的样本，但不立即微调
        self.selected_interesting = interesting
        self.selected_uninteresting = uninteresting

        # 如果预学习已完成，立即微调
        if self.meta_learning_completed:
            self.fine_tune_and_predict()
        else:
            self.status_label.setText("Attraction combinations selected. Waiting for meta-learning to complete...")

    def update_prefs_display(self):
        """更新用户偏好显示"""
        if not self.user_preferences:
            self.prefs_label.setText("User preferences: None")
            return

        sorted_prefs = sorted(self.user_preferences.items(), key=lambda x: x[1], reverse=True)
        prefs_text = ", ".join([f"{k}({v})" for k, v in sorted_prefs[:3]])
        self.prefs_label.setText(f"User preferences: {prefs_text}")

    def on_breadth_changed(self, value):
        """Breadth 滑块调整时实时刷新推荐"""
        self.breadth_value.setText(f"{value} attractions")
        # 使用该城市所有轮次的推荐结果进行筛选
        if hasattr(self, "recommendation_rounds") and self.recommendation_rounds:
            # 这里传入 None，display_recommendations 内部会使用 recommendation_rounds
            self.display_recommendations(None, value)

    def fine_tune_model(self, interesting, uninteresting):
        """微调模型"""
        if hasattr(self, "status_label"):
            self.status_label.setText("Fine-tuning model with user feedback...")
        QApplication.processEvents()

        X = np.array(interesting + uninteresting)
        y = np.array([1] * len(interesting) + [0] * len(uninteresting))

        # 若样本过少，跳过微调以避免 train_test_split 报错
        if len(X) < 2:
            if hasattr(self, "status_label"):
                self.status_label.setText("Too few samples for fine-tuning, skipping fine-tune.")
            try:
                self.append_chat("System", "当前用于微调的样本过少，暂不进行模型微调，但已记录你的偏好。")
            except Exception:
                pass
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 微调模型
        fine_tune_optimizer = Adam(learning_rate=0.004)
        if hasattr(self, "progress_bar"):
            self.progress_bar.setRange(0, 5)

        for epoch in range(5):
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
            # 更新模型监控指标
            if hasattr(self, "model_monitor"):
                self.model_monitor.update_metrics(
                    loss=loss.numpy(),
                    accuracy=train_acc,
                    epoch=epoch + 1,
                    task_info=f"Epoch {epoch + 1}, {len(X_train)} samples"
                )
            if hasattr(self, "progress_bar"):
                self.progress_bar.setValue(epoch + 1)
            QApplication.processEvents()

            # # 更新进度
            # self.progress_bar.setValue(epoch + 1)
            # QApplication.processEvents()

        # 评估模型（右侧监控面板已有准确率信息，这里不再重复显示数值）
        test_accuracy, report = MetaLearning.evaluate_model(self.pre_trained_model, X_test, y_test)

        if hasattr(self, "status_label"):
            self.status_label.setText("Fine-tuning complete!")

        # # 更新最终任务信息
        # self.model_monitor.update_metrics(
        #     loss=loss.numpy(),
        #     accuracy=test_accuracy,
        #     epoch=5,
        #     task_info=f"Completed: Test Acc {test_accuracy:.2%}"
        # )

        # self.status_label.setText(f"Fine-tuning complete! Test accuracy: {test_accuracy:.2%}")
        # self.result_display.append(f"Fine-tuning results:\nAccuracy: {test_accuracy:.2%}")

    def predict_new_patterns(self):
        """预测新的景点组合推荐"""
        if not self.auxiliary_list:
            return

        X_new = np.array(self.auxiliary_list)

        # 使用模型预测每个组合为“感兴趣”(1)的概率
        probs = self.pre_trained_model(X_new, training=False).numpy().squeeze()
        if probs.ndim == 0:
            probs = np.array([probs])

        # 仅保留概率大于 0.5 的组合，相当于“标签为 1”的候选
        positive_mask = probs > 0.5
        positive_indices = np.where(positive_mask)[0]

        # 保存所有 p>0.5 的组合及其概率，供后续按维度筛选 Top-K 使用
        if len(positive_indices) == 0:
            # 本轮没有任何高置信度（p>0.5）的推荐，将空轮次记录下来
            predicted = []
            self.positive_patterns = []
            self.positive_probs = np.array([])
        else:
            positive_probs = probs[positive_indices]
            positive_patterns = [self.auxiliary_list[i] for i in positive_indices]

            # 记录全部 p>0.5 的候选，用于后续“指定维度 Top-K”推荐
            self.positive_patterns = positive_patterns
            self.positive_probs = positive_probs

            # 在这些“标签为 1”的组合中按概率从高到低排序，最多保留前 10 个作为本轮全局 Top-K
            sorted_pos = np.argsort(-positive_probs)
            top_k = min(10, len(sorted_pos))
            top_indices = sorted_pos[:top_k]
            predicted = [positive_patterns[i] for i in top_indices]
        # 记录这一轮的结果（最后一轮）
        self.last_predicted_patterns = predicted

        # 确保有轮次列表
        if not hasattr(self, "recommendation_rounds"):
            self.recommendation_rounds = []

        # 将本轮结果（哪怕为空）加入该城市的轮次列表
        self.recommendation_rounds.append(predicted)
        self.recommendation_round = len(self.recommendation_rounds)

        # 累积到该城市的历史推荐列表（若有其他用途）
        if predicted:
            if not hasattr(self, "all_recommended_patterns"):
                self.all_recommended_patterns = []
            self.all_recommended_patterns.extend(predicted)

        # 无论本轮是否有推荐结果，都立即刷新推荐展示：
        # - 有结果：正常显示组合；
        # - 无结果：该轮会显示 "(No recommended attraction combinations in this round.)"，且保留历史轮次。
        self.display_recommendations(None, None)

    def recommend_topk_for_dimension(self, dim, top_k=10):
        """在所有 p>0.5 的候选中，筛选指定维度的景点组合并输出 Top-K 推荐。"""
        # 确保已存在高置信度候选
        if not hasattr(self, "positive_patterns") or not self.positive_patterns:
            try:
                self.add_message(
                    "system",
                    "There are currently no high-confidence attraction combinations (p>0.5). Please complete a recommendation round first."
                )
            except Exception:
                pass
            return

        if dim < 1 or dim > len(self.feature_names):
            try:
                self.add_message(
                    "system",
                    f"The requested combination size {dim} is out of valid range. Please enter a number between 1 and {len(self.feature_names)}."
                )
            except Exception:
                pass
            return

        # 将所有 p>0.5 的候选按维度进行筛选
        filtered = []
        for pattern, prob in zip(self.positive_patterns, self.positive_probs):
            if sum(pattern) == dim:
                filtered.append((pattern, prob))

        if not filtered:
            try:
                self.add_message(
                    "system",
                    f"There are no recommended attraction combinations with {dim} attractions for the current city."
                )
            except Exception:
                pass
            return

        # 按概率从大到小排序并取前 top_k
        filtered_sorted = sorted(filtered, key=lambda x: -x[1])
        top_k = min(top_k, len(filtered_sorted))
        selected = filtered_sorted[:top_k]

        feature_names = self.feature_names
        lines = []
        for idx, (pattern, prob) in enumerate(selected, start=1):
            feats = [feature_names[j] for j, val in enumerate(pattern) if val == 1]
            lines.append(f"{idx}. {', '.join(feats)}")

        rec_text = "\n\n".join(lines)
        message = (
            f"Based on your current preferences, here are the Top-{top_k} attraction combinations "
            f"with {dim} attractions (p>0.5):\n" + rec_text
        )

        try:
            self.add_message("system", message)
        except Exception:
            # 如果新的聊天气泡界面不可用，则退回到旧的文本聊天区域
            self.append_chat("System", message)

    def display_recommendations(self, predicted, filter_value=None):
        """按照 Breadth 筛选并展示推荐结果"""
        # 所有展示都基于 recommendation_rounds，而不是单次 predicted
        rounds = getattr(self, "recommendation_rounds", [])
        if not rounds:
            # 若存在旧版结果显示控件，则更新之；否则仅通过聊天提示
            if hasattr(self, "result_display"):
                self.result_display.clear()
                self.result_display.append("No recommended attraction combinations found")
                if self.result_display.verticalScrollBar():
                    self.result_display.verticalScrollBar().setValue(0)
            if hasattr(self, "status_label"):
                self.status_label.setText("No recommended combinations found")
            return

        feature_names = self.feature_names

        # 构建完整文本：对每一轮单独应用 Breadth 逻辑，并用分隔符分段
        # 注意：按“最新在上、最早在下”的顺序展示轮次
        result_lines = []
        all_displayed_patterns = []  # 所有轮次在当前 Breadth 下实际显示的组合，用于图谱

        indexed_rounds = list(enumerate(rounds, start=1))
        for round_idx, round_patterns in reversed(indexed_rounds):

            # 对该轮应用 Breadth 筛选逻辑
            filter_note = ""
            if filter_value is None:
                # 不筛选，保留这一轮的全部推荐
                displayed_patterns = round_patterns
            else:
                filtered = [p for p in round_patterns if sum(p) == filter_value]
                if filtered:
                    displayed_patterns = filtered
                else:
                    # 若本轮没有满足 Breadth 的组合，则按照你的要求显示这一轮的全部推荐
                    displayed_patterns = round_patterns
                    filter_note = (
                        f"No combinations with {filter_value} attractions in Round {round_idx}. "
                        "Showing all combinations for this round."
                    )

            # 轮次头部（更简洁的标题样式）
            result_lines.append(f"Round {round_idx} - {self.current_city}")
            result_lines.append("-" * 40)
            if filter_note:
                result_lines.append(filter_note)
            
            if displayed_patterns:
                # 本轮有推荐组合：使用紧凑的编号 + 景点列表形式
                for i, pattern in enumerate(displayed_patterns, start=1):
                    features = [feature_names[j] for j, val in enumerate(pattern) if val == 1]
                    result_lines.append(f"{i}. {', '.join(features)}")

                # 收集本轮展示的所有组合，用于图谱
                all_displayed_patterns.extend(displayed_patterns)
            else:
                # 本轮完全没有任何推荐结果，也要说明
                result_lines.append("(No recommended attraction combinations in this round.)")

            result_lines.append("")
            result_lines.append("")

        # 若存在旧版结果显示控件，则更新之（最新轮次在最上方）
        if hasattr(self, "result_display"):
            self.result_display.clear()
            full_text = "\n".join(result_lines)
            self.result_display.setPlainText(full_text)

            # 视图滚动到最顶部，让最新轮次的标题出现在界面最上方
            if self.result_display.verticalScrollBar():
                self.result_display.verticalScrollBar().setValue(0)

        # 状态栏显示当前 Breadth 下所有轮次的推荐数量（汇总）
        if hasattr(self, "status_label"):
            self.status_label.setText(
                f"Found {len(all_displayed_patterns)} recommended attraction combinations (all rounds under current Breadth)"
            )

        # 同步一份摘要到聊天历史中，让推荐结果也作为对话内容保留
        try:
            short_summary_lines = []
            for line in result_lines:
                # 过滤掉装饰性分隔线，保留关键信息
                if line.startswith("-") or line.startswith("="):
                    continue
                short_summary_lines.append(line)
            summary_text = "\n".join(short_summary_lines).strip()
            if summary_text:
                self.append_chat("System", f"本轮推荐结果：\n{summary_text}")
        except Exception:
            pass


class PatternSelectionDialog(QDialog):
    def __init__(self, patterns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Interesting Attraction Combinations")
        self.setModal(True)
        self.patterns = patterns
        self.checkboxes = []

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # 说明标签
        label = QLabel("Please select attraction combinations you find interesting:")
        label.setStyleSheet("font: bold 12px 'Segoe UI';")

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        # 容器
        container = QWidget()
        container_layout = QVBoxLayout(container)

        # 添加景点组合选择项
        feature_names = ['The Palace Museum', 'Temple of Heaven', 'Summer of Palace', 'Nanluoguxiang',
                         'shichahai', '798 Art District', 'Yonghe Temple', 'National Museum of China', 'Water Cube', 'Mutianyu Great Wall']

        for i, pattern in enumerate(self.patterns):
            features = [feature_names[j] for j, val in enumerate(pattern) if val == 1]
            item_text = f"Combination {i + 1}: {', '.join(features)}"

            group = QGroupBox()
            group.setStyleSheet("""
                QGroupBox {
                    border: 1px solid #E2E8F0;
                    border-radius: 4px;
                    margin-top: 5px;
                }
            """)

            item_layout = QHBoxLayout(group)

            checkbox = QCheckBox(item_text)
            checkbox.setStyleSheet("font: 18px 'Segoe UI';")
            self.checkboxes.append(checkbox)

            # 添加复杂度指示器
            complexity = sum(pattern)
            complexity_bar = QLabel("★" * complexity + "☆" * (10 - complexity))
            complexity_bar.setStyleSheet("color: #8A2BE2; font: 18px 'Segoe UI';")

            item_layout.addWidget(checkbox)
            item_layout.addWidget(complexity_bar)

            container_layout.addWidget(group)

        scroll.setWidget(container)

        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(label)
        layout.addWidget(scroll)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.resize(600, 700)  # 增大选择对话框尺寸

    def get_selected_patterns(self):
        """获取用户选择的景点组合"""
        return [(self.patterns[i], cb.isChecked())
                for i, cb in enumerate(self.checkboxes)]


class ChatComboSelectionWidget(QFrame):
    """嵌入在聊天消息列表中的景点组合选择部件。

    顶部带一个“候选组合数量”调节（作用于后续轮次），下面是当前轮次的候选列表和 Submit 按钮。
    用户点击 Submit 后，通过信号将 (interesting, uninteresting) 传回 MainWindow。
    """

    feedback_submitted = pyqtSignal(object, object)  # (interesting, uninteresting)

    def __init__(self, patterns, feature_names, main_window=None, parent=None):
        super().__init__(parent)
        self.patterns = patterns
        self.feature_names = feature_names
        self.main_window = main_window  # 用于更新后续轮次的 diversity_k
        self.checkboxes = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 顶部一行：当前轮次数量 + 可调节的后续轮次候选数量
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        current_label = QLabel(f"Current candidate combinations: {len(self.patterns)}")
        current_label.setStyleSheet("font: 14px 'Segoe UI';")

        diversity_label = QLabel("Next round count:")
        diversity_label.setStyleSheet("font: 14px 'Segoe UI';")

        self.diversity_spin = QSpinBox()
        self.diversity_spin.setRange(1, 20)
        # 若主窗口存在，则用主窗口当前 diversity_k 作为初始值
        if self.main_window is not None and hasattr(self.main_window, 'diversity_k'):
            self.diversity_spin.setValue(self.main_window.diversity_k)
        else:
            self.diversity_spin.setValue(8)

        self.diversity_spin.valueChanged.connect(self._on_diversity_changed)

        header_layout.addWidget(current_label)
        header_layout.addStretch()
        header_layout.addWidget(diversity_label)
        header_layout.addWidget(self.diversity_spin)

        tip_label = QLabel("Candidate attraction combinations (select the ones you are interested in, then click Submit):")
        tip_label.setStyleSheet("font: 14px 'Segoe UI';")

        # 使用滚动区域承载候选行：恢复为简单的可滚动列表形式
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)

        for i, pattern in enumerate(self.patterns):
            features = [self.feature_names[j] for j, val in enumerate(pattern) if val == 1]
            # 使用更简洁的编号 + 景点名称列表，不再显示 "Combination" 前缀
            item_text = f"{i + 1}. {', '.join(features)}"

            row_frame = QFrame()
            row_layout = QHBoxLayout(row_frame)
            row_layout.setContentsMargins(4, 2, 4, 2)

            cb = QCheckBox(item_text)
            cb.setStyleSheet("font: 14px 'Segoe UI';")
            self.checkboxes.append(cb)

            complexity = sum(pattern)
            stars = "★" * complexity + "☆" * (len(self.feature_names) - complexity)
            complexity_label = QLabel(stars)
            complexity_label.setStyleSheet("color: #8A2BE2; font: 14px 'Segoe UI';")

            row_layout.addWidget(cb)
            row_layout.addWidget(complexity_label)

            container_layout.addWidget(row_frame)

        scroll.setWidget(container)

        self.submit_btn = QPushButton("Submit")
        self.submit_btn.setStyleSheet("background-color: #4A90E2; color: white;")
        self.submit_btn.clicked.connect(self._on_submit)

        layout.addLayout(header_layout)
        layout.addWidget(tip_label)
        layout.addWidget(scroll)
        layout.addWidget(self.submit_btn)

    def _on_diversity_changed(self, value):
        """将顶部 spinbox 的值同步到 MainWindow.diversity_k，用于后续轮次。"""
        if self.main_window is not None and hasattr(self.main_window, 'on_diversity_changed'):
            self.main_window.on_diversity_changed(int(value))

    def _on_submit(self):
        interesting = []
        uninteresting = []
        for pattern, cb in zip(self.patterns, self.checkboxes):
            if cb.isChecked():
                interesting.append(pattern)
            else:
                uninteresting.append(pattern)

        # 触发信号交给 MainWindow 处理
        self.feedback_submitted.emit(interesting, uninteresting)

        # 提交后可以禁用按钮，防止重复提交
        self.submit_btn.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用程序样式
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(74, 144, 226))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())