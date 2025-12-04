import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import binary_crossentropy
from data import pattern_datas_2, support_datas
# from set import support_set_label, query_set_label, binary_patterns


#  from Programme import auxiliary_list
#  from 神经网络 import preprocess_data


def load_data(pattern_data):
    """
    加载数据并划分为训练集和测试集。
    输入:
    - pattern_data: 包含 interesting、uninteresting 和 target 的字典
    - test_size: 测试集比例
    - random_state: 随机种子
    输出:
    - X_train: 训练集特征
    - X_test: 测试集特征
    - y_train: 训练集标签
    - y_test: 测试集标签
    """
    # 将 interesting 和 uninteresting 数组合并为特征 X
    X = np.array(pattern_data["interesting"] + pattern_data["uninteresting"])

    # 生成标签 y
    # interesting 对应标签 1，uninteresting 对应标签 -1
    y = np.array([1] * len(pattern_data["interesting"]) + [-1] * len(pattern_data["uninteresting"]))
    return X, y


def generate_meta_tasks(X, y, num_tasks=5, n_shot=5, n_query=15, k_way=2):
    """
    生成符合MAML要求的元任务数据。
    :param X: 特征数据，列表形式，每个元素是一个长度为 10 的一维数组
    :param y: 标签数据，一维数组，每个元素是样本的标签（1 或 -1）
    :param num_tasks: 每个批次的任务数量
    :param n_shot: 每个类别的支持集样本数
    :param n_query: 每个类别的查询集样本数
    :param k_way: 类别数
    :return: 一个包含 num_tasks 个任务的列表，每个任务为 (support_X, support_y, query_X, query_y)
    """
    meta_tasks = []
    classes = np.unique(y)  # 获取所有类别
    if len(classes) < k_way:
        raise ValueError(f"数据中类别数不足 {k_way} 个")

    X = np.array(X)  # 将 X 转换为 numpy 数组
    y = np.array(y)  # 将 y 转换为 numpy 数组

    for _ in range(num_tasks):
        # 随机选择 k_way 个类别
        selected_classes = np.random.choice(classes, k_way, replace=False)

        support_X, support_y = [], []
        query_X, query_y = [], []

        for cls in selected_classes:
            # 获取当前类别的样本索引
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) < n_shot + n_query:
                raise ValueError(f"类别 {cls} 的样本数不足 {n_shot + n_query} 个")

            # 随机选择 n_shot + n_query 个样本
            selected = np.random.choice(cls_indices, n_shot + n_query, replace=False)
            support = selected[:n_shot]  # 支持集
            query = selected[n_shot:]  # 查询集

            # 添加到任务
            support_X.append(X[support])
            support_y.append(y[support])
            query_X.append(X[query])
            query_y.append(y[query])

        # 合并并打乱顺序
        support_X = np.concatenate(support_X)
        support_y = np.concatenate(support_y)
        query_X = np.concatenate(query_X)
        query_y = np.concatenate(query_y)

        # 打乱顺序
        support_shuffle = np.random.permutation(len(support_X))
        query_shuffle = np.random.permutation(len(query_X))

        meta_tasks.append((
            support_X[support_shuffle],
            support_y[support_shuffle],
            query_X[query_shuffle],
            query_y[query_shuffle]
        ))

    return meta_tasks


class MetaModelWrapper:
    def __init__(self, input_dim, hidden_units=64):
        self.meta_model = self.build_meta_model(input_dim, hidden_units)

    def build_meta_model(self, input_dim, hidden_units):
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(hidden_units, activation='relu'),
            Dense(hidden_units, activation='relu'),
            Dense(1, activation='sigmoid')  # 输出层改为一个节点
        ])
        return model

    def get_weights(self):
        return self.meta_model.get_weights()

    def set_weights(self, weights):
        self.meta_model.set_weights(weights)

    def __call__(self, inputs, training=False):
        return self.meta_model(inputs, training=training)

    @property
    def trainable_variables(self):
        """返回模型的可训练参数"""
        return self.meta_model.trainable_variables


def train_on_batch(meta_model, task_data, inner_optimizer, inner_steps, outer_optimizer=None):
    """
    MAML 单任务训练步骤
    :param meta_model: 元模型
    :param task_data: 任务数据 (support_X, support_y, query_X, query_y)
    :param inner_optimizer: 内循环优化器
    :param inner_steps: 内循环更新步数
    :param outer_optimizer: 外循环优化器（可选）
    :return: 查询集的平均损失和准确率
    """
    support_X, support_y, query_X, query_y = task_data
    # 在 train_on_batch 中转换标签值
    support_y = np.where(support_y == -1, 0, support_y)
    query_y = np.where(query_y == -1, 0, query_y)

    # 将 target 调整为二维
    support_y = np.expand_dims(support_y, axis=-1)
    query_y = np.expand_dims(query_y, axis=-1)

    # 保存元模型的初始权重
    initial_weights = meta_model.get_weights()

    # 内循环：在支持集上更新模型参数
    for _ in range(inner_steps):
        with tf.GradientTape() as tape:
            logits = meta_model(support_X, training=True)
            loss = tf.reduce_mean(binary_crossentropy(support_y, logits))
        grads = tape.gradient(loss, meta_model.trainable_variables)
        inner_optimizer.apply_gradients(zip(grads, meta_model.trainable_variables))

    # 在查询集上计算损失和准确率
    with tf.GradientTape() as tape:
        logits = meta_model(query_X, training=True)
        loss = tf.reduce_mean(binary_crossentropy(query_y, logits))
        acc = tf.reduce_mean(tf.cast((logits > 0.5) == query_y, tf.float32))

    # 恢复元模型的初始权重
    meta_model.set_weights(initial_weights)

    return loss.numpy(), acc.numpy(), tape.gradient(loss, meta_model.trainable_variables)


def maml_train(meta_model, X_scaled, y, num_tasks=5, n_shot=5, n_query=15, k_way=2,
               inner_optimizer=Adam(0.01),
               outer_optimizer=Adam(0.001),
               epochs=50,
               inner_steps=5):
    for epoch in range(epochs):
        meta_tasks = generate_meta_tasks(X_scaled, y, num_tasks, n_shot, n_query, k_way)
        total_loss = 0
        total_acc = 0
        total_grads = [tf.zeros_like(w) for w in meta_model.trainable_variables]  # 初始化外部梯度

        # 每个epoch打乱任务顺序
        np.random.shuffle(meta_tasks)

        for task in meta_tasks:
            # 解包任务数据
            support_X, support_y, query_X, query_y = task

            # 转换数据类型（与原始代码兼容）
            support_y = np.where(support_y == -1, 0, support_y)
            query_y = np.where(query_y == -1, 0, query_y)

            # 执行MAML训练步骤
            loss, acc, grads = train_on_batch(
                meta_model,
                (support_X, support_y, query_X, query_y),
                inner_optimizer,
                inner_steps
            )

            total_loss += loss
            total_acc += acc

            for i, grad in enumerate(grads):
                total_grads[i] += grad

        # 在所有任务上统一更新外部权重
        outer_optimizer.apply_gradients(zip(total_grads, meta_model.trainable_variables))

        # 输出统计信息
        avg_loss = total_loss / len(meta_tasks)
        avg_acc = total_acc / len(meta_tasks)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")


def evaluate_model(model, X_test, y_test):
    """输入:
    - 训练好的模型
    - 预处理后的测试集数据
    输出:
    - accuracy: 准确率
    - report: 分类报告
    """
    # 将标签 -1 转换为 0
    y_test = np.where(y_test == -1, 0, y_test)
    y_pred = (model.predict(X_test) > 0.5).astype(int)  # 将概率转换为类别

    print(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


if __name__ == "__main__":
    # 加载原始数据
    X, y = load_data(pattern_datas_2)

    # 对 X 中的每个样本进行标准化
    X_scaled = [StandardScaler().fit_transform(task.reshape(-1, 1)).flatten() for task in X]

    # 获取输入数据的特征维度
    input_dim = len(X_scaled[0])  # 每个样本的特征数量

    # 初始化元模型
    meta_model = MetaModelWrapper(input_dim, hidden_units=64)

    # 执行MAML训练
    maml_train(
        meta_model,
        X_scaled,
        y,
        num_tasks=10,  # 控制每次迭代的任务数量
        n_shot=20,  # 控制每次迭代的训练数据的数量
        n_query=40,  # 控制每次迭代的测试数据的数量
        k_way=2,  # 提取两个分类
        inner_optimizer=Adam(learning_rate=0.01),
        outer_optimizer=Adam(learning_rate=0.001),
        epochs=15,  # 经过多少次迭代
        inner_steps=5  # 内部训练时参数的更新步长
    )

    # 以上修改第一阶段参数更新的相关超参数------------------------------------------------------------------------------------

    # 最终模型评估--------------------------------------------------------------------------------------------------------
    final_model = meta_model.meta_model

    # 在测试集上微调模型
    print('-------------------------------------------------------------------', '微调阶段')
    fine_tune_epochs = 8
    fine_tune_optimizer = Adam(learning_rate=0.01)

    # 直接使用输入数据中的 interesting 和 uninteresting 部分
    X = np.array(pattern_datas_2["interesting"] + pattern_datas_2["uninteresting"])  # 合并 interesting 和 uninteresting
    y = np.array([1] * len(pattern_datas_2["interesting"]) + [0] * len(pattern_datas_2["uninteresting"]))  # 生成对应的标签

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # 微调模型
    for epoch in range(fine_tune_epochs):
        with tf.GradientTape() as tape:
            logits = final_model(X_train, training=True)  # 使用训练集进行微调
            logits = tf.squeeze(logits, axis=-1)  # 将 logits 从 (num_samples, 1) 变为 (num_samples,)
            loss = tf.reduce_mean(binary_crossentropy(y_train, logits))
        grads = tape.gradient(loss, final_model.trainable_variables)
        fine_tune_optimizer.apply_gradients(zip(grads, final_model.trainable_variables))
        print(f"Fine-tuning Epoch {epoch + 1}/{fine_tune_epochs} | Loss: {loss.numpy():.4f}")

    # 使用 final_model 进行预测
    predictions = final_model.predict(X_test)

    # 将预测结果转换为类别（0 或 1）
    predicted_labels = (predictions > 0.5).astype(int).flatten()

    # 获取 binary_patterns 的最后一位作为真实标签
    # true_labels = [pattern[-1] for pattern in binary_patterns]

    # 计算准确率
    accuracy = np.mean(np.array(predicted_labels) == y_test)

    print(len(predicted_labels), len(y_test))
    # 打印结果
    print(f"Accuracy: {accuracy:.4f}")


