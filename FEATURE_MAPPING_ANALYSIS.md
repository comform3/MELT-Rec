# 特征对应关系分析

## 发现的问题

### 1. **PCPs.py** - 使用字母特征
```python
Feature = {'A', 'B', 'C', 'D', "E", "F", "G", "H", "I", "J"}
```
- 10个特征，用字母表示
- 用于生成元学习预训练数据（support_set_label）

### 2. **program.py** - 使用具体名称特征
```python
self.feature_names = ['University', 'Hospital', 'Library', 'Restaurants',
                     'Cinema', 'Pharmacy', 'Museum', 'Theater', 'Hotel', 'Station']
```
- 10个特征，用具体名称表示
- 用于用户交互和预测

## 对应关系

**目前没有明确的映射关系！**

### 潜在问题

1. **预训练模型和实际数据不匹配**
   - `PCPs.py` 生成的预训练数据使用字母特征（A-J）
   - `program.py` 使用具体名称特征（University, Hospital, ...）
   - 虽然都是10个特征，但**没有明确的对应关系**

2. **特征顺序可能不一致**
   - `PCPs.py`: A, B, C, D, E, F, G, H, I, J
   - `program.py`: University, Hospital, Library, Restaurants, Cinema, Pharmacy, Museum, Theater, Hotel, Station
   - 如果顺序不对应，会导致特征索引错位

## 建议的对应关系

如果假设按字母顺序对应：

```
A → University
B → Hospital  
C → Library
D → Restaurants
E → Cinema
F → Pharmacy
G → Museum
H → Theater
I → Hotel
J → Station
```

但这只是假设，代码中**没有明确的映射定义**。

## 影响

1. **如果特征不对应**：
   - 预训练模型学习的模式可能与实际特征不匹配
   - 模型性能可能下降

2. **如果特征对应但顺序不同**：
   - 二进制向量的索引会错位
   - 预测结果会完全错误

## 建议

1. **在代码中明确定义映射关系**
2. **统一使用相同的特征定义**
3. **或者确保预训练和实际使用使用相同的特征顺序**


