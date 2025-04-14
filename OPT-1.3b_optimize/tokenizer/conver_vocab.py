import json

def swap_key_value(json_file, output_file):
    # 1. 打开并读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 交换键值对，确保值是唯一的以成为新键
    swapped_data = {v: k for k, v in data.items()}

    # 3. 将新的字典写入到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(swapped_data, f, ensure_ascii=False, indent=4)

    print(f"键值对已互换并保存到: {output_file}")

# 调用函数，输入和输出文件路径
input_json = '/home/zjnyly/Cyclotron0913/vocab.json'  # 替换为你的输入 JSON 文件路径
output_json = '/home/zjnyly/Cyclotron0913/vocab_rev.json'  # 替换为你想要保存的输出文件路径
swap_key_value(input_json, output_json)