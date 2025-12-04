import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import binary_crossentropy


class Nnetwork:
    @staticmethod
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

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    @staticmethod
    def train_model(X_train, y_train, input_dim, hidden_units=64, epochs=5, learning_rate=0.01):
        """
        训练模型
        输入:
        - X_train: 训练集特征
        - y_train: 训练集标签
        - input_dim: 输入特征维度
        - hidden_units: 隐藏层神经元数量
        - epochs: 训练轮数
        - learning_rate: 学习率
        输出:
        - 训练好的模型
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

        return model

    @staticmethod
    def predict_with_model(model, X):
        """
        使用训练好的模型进行预测
        输入:
        - model: 训练好的模型
        - X: 需要预测的数据
        输出:
        - 预测结果（类别：0 或 1）
        """
        y_pred = (model.predict(X) > 0.5).astype(int)
        return y_pred


# if __name__ == "__main__":
#     # 加载新数据集 pattern_data_2
#     X_new = np.array(pattern_datas["interesting"] + pattern_datas["uninteresting"])
#     y_new = np.array([1] * len(pattern_datas["interesting"]) + [0] * len(pattern_datas["uninteresting"]))
#
#     # 获取输入数据的特征维度
#     input_dim = len(X_new[0])  # 每个样本的特征数量
#
#     # 使用传统方法训练模型
#     print("\n传统方法训练模型：")
#     model = Nnetwork.train_model(
#         X_new, y_new, input_dim, hidden_units=64, epochs=3, learning_rate=0.01
#     )
#
#     # 使用训练好的模型对 auxiliary_list 中的数据进行预测
#     if auxiliary_list:
#         X_auxiliary = np.array(auxiliary_list)
#         y_pred = Nnetwork.predict_with_model(model, X_auxiliary)
#         print("\n对 auxiliary_list 的预测结果：")
#         print(y_pred)
#     else:
#         print("auxiliary_list 为空，无法进行预测。")
