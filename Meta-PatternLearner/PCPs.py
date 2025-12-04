import itertools
import random

# 定义特征集合（10 维位置，不区分城市，这里只表示位置索引）
Feature = {'A', 'B', 'C', 'D', "E", "F", "G", "H", "I", "J"}

# 用于存储所有组合的列表
all_combinations = []

# 生成所有可能的组合（长度>=2）
for r in range(2, len(Feature) + 1):
    combinations = itertools.combinations(Feature, r)
    for combo in combinations:
        all_combinations.append(''.join(combo))

# 将模式转换为0和1的形式
binary_patterns = []
for pattern in all_combinations:
    binary_pattern = [1 if letter in pattern else 0 for letter in Feature]
    binary_patterns.append(binary_pattern)

# 随机挑选 X 个模式作为“支持集候选”，并从总模式集中删除
X = 3000
supprot_set_patterns = []
support_set_binary_patterns = []

for _ in range(X):
    if not all_combinations:
        print("No more patterns available.")
        break
    random_index = random.randrange(len(all_combinations))
    supprot_set_patterns.append(all_combinations[random_index])
    support_set_binary_patterns.append(binary_patterns[random_index])
    del all_combinations[random_index]
    del binary_patterns[random_index]

# 再从剩余模式中挑选 X 个作为 query set，并从总模式集中删除
X = 20
query_set_patterns = []
query_set_binary_patterns = []

for _ in range(X):
    if not all_combinations:
        print("No more patterns available.")
        break
    random_index = random.randrange(len(all_combinations))
    query_set_patterns.append(all_combinations[random_index])
    query_set_binary_patterns.append(binary_patterns[random_index])
    del all_combinations[random_index]
    del binary_patterns[random_index]

# 北京支持集：最后一位为 1 判定感兴趣
support_set_label_beijing = {
    'interesting': [],
    'uninteresting': [],
    'target': [[], []]
}

# 上海支持集：倒数第二位为 1 判定感兴趣
support_set_label_shanghai = {
    'interesting': [],
    'uninteresting': [],
    'target': [[], []]
}
# query set（共用一份，按北京规则）
query_set_label = {
    'interesting': [],
    'uninteresting': [],
    'target': [[], []]
}

# 处理 support set：
# - 北京：根据最后一位是否为 1 判定兴趣
# - 上海：根据倒数第二位是否为 1 判定兴趣
for binary_pattern in support_set_binary_patterns:
    # 北京规则（最后一位）
    if binary_pattern[-1] == 1:
        support_set_label_beijing['interesting'].append(binary_pattern)
        support_set_label_beijing['target'][0].append(1)
    else:
        support_set_label_beijing['uninteresting'].append(binary_pattern)
        support_set_label_beijing['target'][1].append(0)

    # 上海规则（倒数第二位）
    if binary_pattern[-2] == 1:
        support_set_label_shanghai['interesting'].append(binary_pattern)
        support_set_label_shanghai['target'][0].append(1)
    else:
        support_set_label_shanghai['uninteresting'].append(binary_pattern)
        support_set_label_shanghai['target'][1].append(0)

# 处理 query set（沿用北京“最后一位”规则）
for binary_pattern in query_set_binary_patterns:
    if binary_pattern[-1] == 1:
        query_set_label['interesting'].append(binary_pattern)
        query_set_label['target'][0].append(1)
    else:
        query_set_label['uninteresting'].append(binary_pattern)
        query_set_label['target'][1].append(0)

# 噪声：目前不开启，保持 0
Z = 0
noise_indices = random.sample(range(len(binary_patterns)), Z) if binary_patterns else []
noise_patterns = [binary_patterns[i] for i in noise_indices]

for binary_pattern in noise_patterns:
    if random.choice([True, False]):
        support_set_label_beijing['interesting'].append(binary_pattern)
        support_set_label_beijing['target'][0].append(1)
    else:
        support_set_label_beijing['uninteresting'].append(binary_pattern)
        support_set_label_beijing['target'][1].append(0)

print("[PCPs] Beijing interesting:", len(support_set_label_beijing['interesting']),
      "uninteresting:", len(support_set_label_beijing['uninteresting']))
print("[PCPs] Shanghai interesting:", len(support_set_label_shanghai['interesting']),
      "uninteresting:", len(support_set_label_shanghai['uninteresting']))