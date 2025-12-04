import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import binary_crossentropy
from data import pattern_datas_2
from 神经网络 import evaluate_model


def traditional_train_and_evaluate(X_train, y_train, X_test, y_test, input_dim, hidden_units=64, epochs=5, learning_rate=0.01):
    """
    传统训练和评估方法：从头开始训练一个模型，并在测试集上评估。
    输入:
    - X_train: 训练集特征
    - y_train: 训练集标签
    - X_test: 测试集特征
    - y_test: 测试集标签
    - input_dim: 输入特征维度
    - hidden_units: 隐藏层神经元数量
    - epochs: 训练轮数
    - learning_rate: 学习率
    输出:
    - test_accuracy: 测试集准确率
    - report: 分类报告
    """
    # 构建模型
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(hidden_units, activation='relu'),
        Dense(hidden_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # 编译模型
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=binary_crossentropy, metrics=['accuracy'])

    # 训练模型
    print("从头开始训练模型...")
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            logits = model(X_train, training=True)
            logits = tf.squeeze(logits, axis=-1)
            loss = tf.reduce_mean(binary_crossentropy(y_train, logits))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"Training Epoch {epoch + 1}/{epochs} | Loss: {loss.numpy():.4f}")

    # 评估模型
    test_accuracy, report = evaluate_model(model, X_test, y_test)
    return test_accuracy, report


if __name__ == "__main__":

    # 加载新数据集 pattern_data_2
    X_new = np.array(pattern_datas_2["interesting"] + pattern_datas_2["uninteresting"])
    y_new = np.array([1] * len(pattern_datas_2["interesting"]) + [0] * len(pattern_datas_2["uninteresting"]))

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.8, random_state=42)

    # 获取输入数据的特征维度
    input_dim = len(X_train[0])  # 每个样本的特征数量

    # 使用传统方法训练和评估模型
    print("\n传统方法训练和评估：")
    traditional_accuracy, traditional_report = traditional_train_and_evaluate(
        X_train, y_train, X_test, y_test, input_dim, hidden_units=64, epochs=3, learning_rate=0.01
    )
    print(f"\n传统方法测试准确率: {traditional_accuracy:.4f}")
    print("传统方法分类报告:\n", traditional_report)