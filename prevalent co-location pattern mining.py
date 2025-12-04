import math
from math import sqrt
import itertools


def Compute_number(feature, instance):
    Instance = {}
    for elem in feature:
        number_instance = 0
        for i in instance:
            if i[0][0] == elem:
                number_instance += 1
        Instance[elem] = number_instance
    return Instance


#  计算出所有的实例邻近关系
def computed_neigh(instance_1, instance_2, d):
    x = instance_1[1] - instance_2[1]
    y = instance_1[2] - instance_2[2]
    distance_12 = sqrt(x ** 2 + y ** 2)
    if distance_12 <= d:
        return True
    else:
        return False


def grid_method(i, d):
    """
    采用网格法计算每一个实例的邻近结点，每一个实例只用检查本身所在网格以及与它相邻的八个方向的网格
    :param i: 实例集
    :param d: 网格变成（距离阈值）
    :return: Ns，PNs，SNs
    """
    # 找到整个网格的范围，对网格进行划分，并将实例分配进对应的网格
    x_min = min(x[1] for x in i)
    y_min = min(x[2] for x in i)
    hash_grid = {}
    for elem in i:
        x_order = math.ceil((elem[1] - x_min) / d)
        y_order = math.ceil((elem[2] - y_min) / d)
        if x_order == 0:
            x_order += 1
        if y_order == 0:
            y_order += 1
        hash_grid.setdefault((x_order, y_order), []).append(elem)

    # 根据划分的网格计算每个实例的邻近关系
    Ns = {}
    computed_neigh = lambda x, y, d: (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2 <= d ** 2
    for elem_hash, grid_instances in hash_grid.items():
        for elem_list in grid_instances:
            Ns.setdefault(elem_list[0], [])
            for x in grid_instances:
                if x[0][0] != elem_list[0][0] and computed_neigh(x, elem_list, d):
                    Ns[elem_list[0]].append(x[0])

            # 计算相邻的八个方向
            for delta_x in range(-1, 2):
                for delta_y in range(-1, 2):
                    if delta_x == delta_y == 0:
                        continue
                    adjacent_grid = (elem_hash[0] + delta_x, elem_hash[1] + delta_y)
                    if adjacent_grid in hash_grid:
                        for elem_xy in hash_grid[adjacent_grid]:
                            if elem_xy[0][0] != elem_list[0][0] and computed_neigh(elem_xy, elem_list, d):
                                Ns[elem_list[0]].append(elem_xy[0])
    return Ns


def digui(can_key, inter, nis, A):
    # print(can_key, "can_key")
    # print(inter, "inter")
    if len(inter) == 0:
        A.append(can_key)
        return
    if len(inter) == 1:
        A.append(can_key + [inter[0]])
        return
    pre_inter = inter.copy()
    for elem in inter:
        pre_inter.pop(0)
        new_can_key = can_key + [elem]
        new_inter = sorted(list(set(nis[elem]) & set(pre_inter)))
        digui(new_can_key, new_inter, nis, A)


def is_element_in_nested_list(lst, element):
    for sublist in lst:
        if sublist is not None:
            if element in sublist:
                return True
    return False


def Enum_Cliques(nis):
    clique = {}
    for key in nis:
        clique[key] = []
        flag_list = nis[key].copy()

        while len(flag_list) != 0:
            Can_key = [flag_list[0]]
            inter = sorted(list(set(nis[flag_list[0]]) & set(flag_list)))
            flag_list.remove(flag_list[0])

            A = []
            digui(Can_key, inter, nis, A)

            if len(A) != 0:
                for elem in A:
                    clique[key].append(elem)
            else:
                s = Can_key
                if not is_element_in_nested_list(clique[key], Can_key[0]):
                    clique[key].append(s)
                if Can_key != s:
                    clique[key].append(s)
    return clique


#  生成哈希表
def con_hash_table(clique):
    hash_table = {}  # 哈希表
    for key in clique:
        for clique_list in clique[key]:
            if clique_list is None:
                continue

            # 构建哈希键
            hash_key = key[0] + ''.join(elem[0] for elem in clique_list)
            hash_key = ''.join(sorted(hash_key))

            # 创建哈希表中的内层哈希表
            if hash_key not in hash_table:
                hash_table[hash_key] = {}
                hash_table[hash_key][key[0]] = [key]
                for elem in clique_list:
                    hash_table[hash_key][elem[0]] = [elem]
            else:
                # 添加关键字实例和特征实例
                if key[0] not in hash_table[hash_key]:
                    hash_table[hash_key][key[0]] = [key]
                else:
                    if key not in hash_table[hash_key][key[0]]:
                        hash_table[hash_key][key[0]].append(key)
                for elem in clique_list:
                    if elem[0] not in hash_table[hash_key]:
                        hash_table[hash_key][elem[0]] = [elem]
                    else:
                        if elem not in hash_table[hash_key][elem[0]]:
                            hash_table[hash_key][elem[0]].append(elem)
    return hash_table


def Find_prevalent_patterns(hash_table, feature, min_prev, instance_num):
    # 用于存储所有子集的集合
    all_subsets = set()

    # 生成所有可能的子集
    for pattern in hash_table.keys():
        for r in range(2, len(pattern) + 1):
            subsets = itertools.combinations(pattern, r)
            for subset in subsets:
                all_subsets.add(''.join(subset))

    # 将集合转换为列表
    unique_subsets = list(all_subsets)

    pattern_hash = {}  # 表实例
    for patterns in unique_subsets:
        pattern_hash[patterns] = {}

    for Can_pattern in pattern_hash.keys():
        for elem_1 in Can_pattern:
            pattern_hash[Can_pattern][elem_1] = []

        for hash_key in hash_table.keys():
            letters_set = set(Can_pattern)
            if all(letter in hash_key for letter in letters_set):  # 证明这个hash键里包含这个候选的所有字符
                for elem in Can_pattern:
                    for instance in hash_table[hash_key][elem]:
                        if instance not in pattern_hash[Can_pattern][elem]:
                            pattern_hash[Can_pattern][elem].append(instance)

    print("表实例寻找完成")

    PI_hash = {}
    for Can_pattern in pattern_hash.keys():
        PI_hash[Can_pattern] = 0
        min_PR = 1
        for elem_1 in Can_pattern:
            PR = len(pattern_hash[Can_pattern][elem_1]) / instance_num[elem_1]
            if PR < min_PR:
                min_PR = PR
        PI_hash[Can_pattern] = min_PR

    prevalent_patterns = []
    for key in PI_hash:
        if PI_hash[key] >= min_prev:
            prevalent_patterns.append(key)

    return prevalent_patterns


if __name__ == "__main__":
    f = open(r"C:\Users\12521\Desktop\应用期刊\Beijing_.csv", "r",
             encoding="UTF-8")  # AA.text BB.text CC.text
    Instance = []
    for line in f:
        temp_2 = []
        temp = line.strip().split(",")
        if temp != [''] and temp != ['Feature', 'Instance', 'LocX', 'LocY', 'Checkin']:
            s = temp[0] + temp[1]
            temp_2.append(s)
            temp_2.append(float(temp[2]))
            temp_2.append(float(temp[3]))
            Instance.append(temp_2)
    f.close()
    Feature = {'A', 'B', 'C', 'D', "E", "F", "G", "H", "I", "J", 'K', 'L', 'M', 'N', 'O'}
    Min_prev = 0.1
    D = 1000  # 距离阈值越大生成得邻近关系越多

    Instance_num = Compute_number(Feature, Instance)
    NIS = grid_method(Instance, D)
    Clique = Enum_Cliques(NIS)
    Hash_table = con_hash_table(Clique)
    print(Hash_table)
    Pre_pattern = Find_prevalent_patterns(Hash_table, Feature, Min_prev, Instance_num)
    print(Pre_pattern)
    print(len(Pre_pattern))
