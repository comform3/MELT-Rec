from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import numpy as np


def load_data(test_size=0.5, random_state=42):
    """输入: 无
    输出:
    - X_train: 训练集特征
    - X_test: 测试集特征
    - y_train: 训练集标签
    - y_test: 测试集标签
    """
    data = pattern_data
    X, y = data["data"], data["target"]
    # 将标签 0 转换为 -1
    y = np.where(y == 0, -1, y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess_data(X_train, X_test):
    """输入:
    - X_train: 原始训练集特征
    - X_test: 原始测试集特征
    输出:
    - X_train_scaled: 标准化后的训练集特征
    - X_test_scaled: 标准化后的测试集特征
    - scaler: 标准化器对象（用于后续新数据）
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train, input_dim, hidden_units=128, learning_rate=0.001, epochs=50, batch_size=32):
    """输入:
    - X_train: 预处理后的训练集特征（需为 float32 类型的二维数组）
    - y_train: 训练集标签（二分类需为 0/1 的 int 或 float）
    - input_dim: 输入特征的维度（必须与 X_train.shape[1] 一致）
    """
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    # 将标签 -1 转换为 0
    y_train = np.where(y_train == -1, 0, y_train)

    model = Sequential()
    # 新版正确写法：用 Input 层定义输入维度
    model.add(Input(shape=(input_dim,)))  # ✅ 替代旧版 input_dim 参数
    model.add(Dense(hidden_units, activation='relu'))  # 第一个隐藏层
    model.add(Dense(hidden_units, activation='relu'))  # 第二个隐藏层
    model.add(Dense(1, activation='sigmoid'))  # 输出层

    # 编译模型
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=0)
    return model


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


def predict_new_sample(model, scaler, new_sample):
    """输入:
    - 训练好的模型
    - 之前生成的标准化器
    - 新样本数据（未处理的原始格式）
    输出: 预测类别
    """
    print(scaler)
    print(new_sample)
    scaled_sample = scaler.transform([new_sample])
    prediction = (model.predict(scaled_sample) > 0.5).astype(int)[0]  # 将概率转换为类别

    # 将输出 0 转换回 -1
    prediction = -1 if prediction == 0 else 1
    return prediction


# 使用示例 ------------------------------------------------------------------------
if __name__ == "__main__":
    # 输入数据
    X_train, X_test, y_train, y_test = load_data()

    # 数据预处理
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

    # 训练模型
    input_dim = X_train_scaled.shape[1]  # 输入特征的维度

    model = train_model(X_train_scaled, y_train, input_dim, hidden_units=64, learning_rate=0.001, epochs=50,
                        batch_size=32)

    print(type(X_test_scaled), type(y_test))
    # 评估输出
    accuracy, report = evaluate_model(model, X_test_scaled, y_test)
    print(f"模型准确率: {accuracy:.4f}\n分类报告:\n{report}")

    # 新样本预测示例
    new_samples = [
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 1, 0]
    ]  # 假设这是一组新样本（原始未标准化数据）
    for sample in new_samples:
        prediction = predict_new_sample(model, scaler, sample)
        print(f"新样本预测结果: {'用户感兴趣' if prediction == 1 else '用户不感兴趣'}")
