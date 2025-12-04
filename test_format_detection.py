"""
测试CSV格式检测和转换功能
"""
import sys
import os

# 添加Meta-PatternLearner到路径
sys.path.insert(0, 'Meta-PatternLearner')

def test_format_detection():
    """测试格式检测逻辑"""
    print("="*60)
    print("测试CSV格式检测")
    print("="*60)
    
    # 模拟格式检测函数
    def detect_format(csv_data):
        if not csv_data or len(csv_data) == 0:
            return "binary_vector"
        
        first_row = csv_data[0]
        
        try:
            int(first_row[0])
            if all(x in ['0', '1'] for row in csv_data[:5] for x in row):
                return "binary_vector"
            else:
                return "mixed"
        except ValueError:
            try:
                float(first_row[1]) if len(first_row) > 1 else None
                return "text_float"
            except:
                return "mixed"
    
    # 测试案例1: 二进制向量
    test1 = [
        ['1', '0', '1', '0', '1'],
        ['0', '1', '0', '1', '0'],
        ['1', '1', '0', '0', '1']
    ]
    result1 = detect_format(test1)
    print(f"\n测试1 - 二进制向量:")
    print(f"  数据: {test1[0]}")
    print(f"  检测结果: {result1}")
    print(f"  ✅ PASS" if result1 == "binary_vector" else f"  ❌ FAIL")
    
    # 测试案例2: 文本+浮点数
    test2 = [
        ['Restaurant', '39.978658', '116.307898'],
        ['Cinema', '39.979228', '116.305721'],
        ['Hospital', '39.976179', '116.307838']
    ]
    result2 = detect_format(test2)
    print(f"\n测试2 - 文本+浮点数:")
    print(f"  数据: {test2[0]}")
    print(f"  检测结果: {result2}")
    print(f"  ✅ PASS" if result2 == "text_float" else f"  ❌ FAIL")
    
    # 测试案例3: 混合格式
    test3 = [
        ['5', '3', '2', '8'],
        ['12', '7', '4', '9']
    ]
    result3 = detect_format(test3)
    print(f"\n测试3 - 混合格式:")
    print(f"  数据: {test3[0]}")
    print(f"  检测结果: {result3}")
    print(f"  ✅ PASS" if result3 == "mixed" else f"  ❌ FAIL")

def test_text_float_conversion():
    """测试文本+浮点数转换"""
    print("\n" + "="*60)
    print("测试文本+浮点数转换")
    print("="*60)
    
    # 模拟数据
    csv_data = [
        ['Restaurant', '39.97', '116.30'],
        ['Hospital', '39.97', '116.305'],  # 接近Restaurant
        ['Cinema', '39.98', '116.30']      # 较远
    ]
    
    # 提取特征类型
    feature_types = list(set(row[0] for row in csv_data))
    feature_types.sort()
    
    print(f"\n提取的特征类型: {feature_types}")
    
    # 存储原始数据
    raw_data = []
    for row in csv_data:
        feature = row[0]
        coords = [float(row[1]), float(row[2])]
        raw_data.append((feature, coords))
    
    # 计算共现
    distance_threshold = 0.01
    patterns_dict = {}
    
    for i, (feature, coords) in enumerate(raw_data):
        nearby_features = set([feature])
        
        for j, (other_feature, other_coords) in enumerate(raw_data):
            if i != j:
                dist = ((coords[0] - other_coords[0])**2 + 
                       (coords[1] - other_coords[1])**2)**0.5
                if dist <= distance_threshold:
                    nearby_features.add(other_feature)
        
        pattern = [1 if feat in nearby_features else 0 for feat in feature_types]
        pattern_tuple = tuple(pattern)
        
        if pattern_tuple not in patterns_dict and sum(pattern) >= 2:
            patterns_dict[pattern_tuple] = pattern
        
        print(f"\n{feature} @ ({coords[0]}, {coords[1]})")
        print(f"  附近特征: {nearby_features}")
        print(f"  模式向量: {pattern}")
    
    patterns = list(patterns_dict.values())
    print(f"\n生成的唯一模式数: {len(patterns)}")
    print(f"模式列表:")
    for i, p in enumerate(patterns):
        features = [feature_types[j] for j, val in enumerate(p) if val == 1]
        print(f"  {i+1}. {p} → {', '.join(features)}")

def test_file_loading():
    """测试实际文件加载"""
    print("\n" + "="*60)
    print("测试实际文件加载")
    print("="*60)
    
    import csv
    
    # 测试文件1: test_text_float.csv
    if os.path.exists('test_text_float.csv'):
        print("\n✅ 找到 test_text_float.csv")
        with open('test_text_float.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader)
        print(f"  行数: {len(data)}")
        print(f"  示例: {data[0]}")
        
        # 统计特征类型
        features = set(row[0] for row in data)
        print(f"  特征类型数: {len(features)}")
        print(f"  特征: {sorted(features)}")
    else:
        print("\n❌ 未找到 test_text_float.csv")
    
    # 测试文件2: Beijing_haidian.csv
    if os.path.exists('Beijing_haidian.csv'):
        print("\n✅ 找到 Beijing_haidian.csv")
        with open('Beijing_haidian.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader)
        print(f"  行数: {len(data)}")
        print(f"  示例: {data[0]}")
        
        features = set(row[0] for row in data)
        print(f"  特征类型数: {len(features)}")
        print(f"  特征: {sorted(features)}")
    else:
        print("\n❌ 未找到 Beijing_haidian.csv")

if __name__ == "__main__":
    test_format_detection()
    test_text_float_conversion()
    test_file_loading()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


