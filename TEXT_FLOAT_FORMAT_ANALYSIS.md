# Text_Float格式数据流程分析

## 完整工作流程：从CSV数据到用户兴趣预测

### 阶段1：数据加载与格式转换

#### 1.1 CSV文件读取
**输入示例** (`test_text_float.csv`):
```
Restaurants,39.978658,116.307898
Cinema,39.979228,116.305721
Hospital,39.976179,116.307838
School,39.976984,116.320344
...
```

**格式特点**：
- 第1列：特征类型（文本，如Restaurants, Cinema, Hospital等）
- 第2列：经度（浮点数）
- 第3列：纬度（浮点数）

#### 1.2 格式检测 (`detect_format`)
```python
# 检测逻辑：
1. 检查第一列是否为文本（非数字）
2. 检查第二列是否为浮点数
3. 如果满足 → 识别为 "text_float" 格式
```

#### 1.3 数据转换 (`process_text_float_format`)

**步骤1：提取特征类型**
```python
feature_types = ['Cinema', 'Hospital', 'Restaurants', 'School', 'Station', 'Theater']
# 自动去重并排序
```

**步骤2：存储原始坐标数据**
```python
self.raw_data = [
    ('Restaurants', [39.978658, 116.307898]),
    ('Cinema', [39.979228, 116.305721]),
    ('Hospital', [39.976179, 116.307838]),
    ...
]
```

**步骤3：基于空间共现生成模式**

**核心算法**：
- **距离阈值**：`distance_threshold = 0.01`（约1公里）
- **共现判断**：对于每个数据点，找出距离阈值内的所有其他特征

**示例计算**：
```
点1: Restaurants (39.978658, 116.307898)
点2: Cinema (39.979228, 116.305721)
距离 = sqrt((39.978658-39.979228)² + (116.307898-116.305721)²)
     = sqrt(0.000325 + 0.0000047) ≈ 0.018

如果距离 ≤ 0.01 → 认为共现
```

**步骤4：生成二进制向量模式**

对于每个位置，生成一个二进制向量：
```python
# 假设特征类型顺序：['Cinema', 'Hospital', 'Restaurants', 'School', 'Station', 'Theater']
# 如果某个位置附近有：Restaurants, Cinema, Hospital
# 则生成向量：[1, 1, 1, 0, 0, 0]
```

**去重逻辑**：
- 只保留包含至少2个特征的模式（`sum(pattern) >= 2`）
- 使用元组作为字典键去重

**最终输出**：
```python
self.auxiliary_list = [
    [1, 1, 1, 0, 0, 0],  # Restaurants + Cinema + Hospital
    [0, 0, 1, 1, 0, 0],  # Restaurants + School
    [1, 0, 0, 0, 1, 0],  # Cinema + Station
    ...
]
```

---

### 阶段2：元学习预训练（后台进行）

**使用数据**：`PCPs.py` 生成的 `support_set_label`（3000个预定义模式）

**训练过程**：
1. MAML算法在支持集上快速适应
2. 在查询集上评估并更新元参数
3. 训练15个epoch，生成预训练模型

**输出**：`self.pre_trained_model` - 具备快速适应能力的元模型

---

### 阶段3：用户交互与反馈

#### 3.1 模式选择 (`start_interaction`)

**选择策略**：
```python
num_patterns = 5 + diversity_slider.value() * 2
# diversity_slider = 5 → 选择 5 + 5*2 = 15个模式
selected = random.sample(self.auxiliary_list, num_patterns)
```

**显示给用户**：
- 按复杂度排序（模式中1的个数）
- 显示每个模式包含的特征
- 用户通过复选框选择感兴趣的模式

**示例显示**：
```
Pattern 1: Restaurants, Cinema, Hospital  [★★★☆☆☆☆☆☆☆]
Pattern 2: School, Station                 [★★☆☆☆☆☆☆☆☆]
Pattern 3: Theater, Cinema                  [★★☆☆☆☆☆☆☆☆]
...
```

#### 3.2 用户反馈处理 (`process_user_feedback`)

**输入**：用户选择的模式列表
```python
selected_patterns = [
    ([1, 1, 1, 0, 0, 0], True),   # 用户选择了这个
    ([0, 0, 1, 1, 0, 0], False),  # 用户没选择
    ...
]
```

**分类**：
```python
interesting = [所有用户勾选的模式]
uninteresting = [所有用户未勾选的模式]
```

**用户偏好统计**：
```python
# 统计每个特征在感兴趣模式中出现的次数
user_preferences = {
    'Restaurants': 3,
    'Cinema': 2,
    'Hospital': 1,
    ...
}
```

---

### 阶段4：模型微调

#### 4.1 准备训练数据 (`fine_tune_model`)

```python
X = np.array(interesting + uninteresting)
# 例如：[[1,1,1,0,0,0], [0,0,1,1,0,0], ...]

y = np.array([1, 1, ..., 0, 0, ...])
# 1 = interesting, 0 = uninteresting
```

#### 4.2 微调过程

**训练流程**：
1. 将数据分为训练集（70%）和测试集（30%）
2. 使用预训练模型进行5个epoch的微调
3. 学习率：0.004（较小，因为模型已经预训练）
4. 损失函数：二元交叉熵
5. 优化器：Adam

**关键代码**：
```python
for epoch in range(5):
    with tf.GradientTape() as tape:
        logits = self.pre_trained_model(X_train, training=True)
        loss = binary_crossentropy(y_train, logits)
    
    grads = tape.gradient(loss, self.pre_trained_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, ...))
```

**结果**：模型学会了用户的偏好模式

---

### 阶段5：预测用户兴趣

#### 5.1 批量预测 (`predict_new_patterns`)

**输入**：`self.auxiliary_list` 中的所有剩余模式

**预测过程**：
```python
X_new = np.array(self.auxiliary_list)  # 所有未标注的模式
y_pred = model.predict(X_new)          # 预测概率

# 阈值判断
predicted = [pattern for pattern, pred in zip(auxiliary_list, y_pred) 
             if pred == 1]  # 概率 > 0.5 认为是感兴趣
```

#### 5.2 结果展示

**文本显示**：
```
Predicted interesting patterns:
Pattern 1: Restaurants, Cinema, Hospital
Pattern 2: School, Station, Theater
...
```

**可视化**：
- 相似性图谱：显示预测模式之间的关系
- 基于Jaccard相似度构建网络图
- 节点大小表示模式复杂度

---

## 关键问题分析

### 问题1：特征名称不匹配

**发现的问题**：
```python
# process_text_float_format 中使用的特征名（动态）
self.feature_names = ['Cinema', 'Hospital', 'Restaurants', ...]

# 但在 process_user_feedback 和 predict_new_patterns 中使用的是硬编码
feature_names = ['University', 'Hospital', 'Library', 'Restaurants', ...]
```

**影响**：
- 用户偏好统计可能不准确
- 预测结果显示的特征名可能错误

### 问题2：auxiliary_list未更新

**当前行为**：
- 用户标注的模式仍然保留在 `auxiliary_list` 中
- 预测时会包含已标注的模式

**建议修复**：
```python
# 在 process_user_feedback 中添加
for pattern in interesting + uninteresting:
    if pattern in self.auxiliary_list:
        self.auxiliary_list.remove(pattern)
```

---

## 完整流程图

```
CSV文件 (text_float格式)
    ↓
格式检测 → text_float
    ↓
提取特征类型 → ['Cinema', 'Hospital', ...]
    ↓
计算空间共现 → 基于距离阈值(0.01)
    ↓
生成二进制向量 → auxiliary_list
    ↓
用户交互 → 选择感兴趣的模式
    ↓
模型微调 → 使用用户反馈
    ↓
预测剩余模式 → 找出用户可能感兴趣的模式
    ↓
可视化展示 → 相似性图谱 + 文本列表
```

---

## 数据转换示例

**原始数据**：
```
Restaurants,39.978658,116.307898
Cinema,39.979228,116.305721
Hospital,39.976179,116.307838
```

**转换后**：
```
特征类型: ['Cinema', 'Hospital', 'Restaurants', 'School', 'Station', 'Theater']

模式1: [1, 1, 1, 0, 0, 0]  # Restaurants + Cinema + Hospital (距离近)
模式2: [0, 1, 0, 1, 0, 0]  # Hospital + School (距离近)
...
```

**用户选择模式1** → 模型学习：Restaurants + Cinema + Hospital 的组合是用户感兴趣的

**预测结果**：系统会推荐包含类似特征组合的其他模式


