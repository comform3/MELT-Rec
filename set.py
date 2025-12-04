import csv
import random


def generate_unique_vectors(num_rows):
    unique_vectors = set()  # 用集合存储向量，确保唯一性

    while len(unique_vectors) < num_rows:
        # 生成一个10维向量，每个元素是0或1
        vector = [random.randint(0, 1) for _ in range(10)]

        # 检查是否至少有两个1，并且是否已经存在
        if sum(vector) >= 2 and tuple(vector) not in unique_vectors:
            unique_vectors.add(tuple(vector))

    # 转换回列表形式
    return [list(vector) for vector in unique_vectors]


def shuffle_data(data):
    random.shuffle(data)
    return data


def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


if __name__ == "__main__":
    num_rows = 436
    data = generate_unique_vectors(num_rows)

    for i in range(num_rows):
        if len(data[i]) <2:
            print(data[i])

    # 打乱数据顺序
    shuffled_data = shuffle_data(data)

    # 写入CSV文件
    write_to_csv('prevalent patterns.csv', shuffled_data)

    print(f"生成了 {num_rows} 行数据，并已写入 prevalent patterns.csv。")