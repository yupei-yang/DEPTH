from datasets import Dataset, DatasetDict
import json

# 读取原始数据
with open('dataset/SciERC/train_data/json_files/processed/SciERC-RM-full.json') as f:
    data = json.load(f)  # 直接加载整个JSON数组

# 转换为Dataset格式
dataset = Dataset.from_list(data)  # 直接使用加载的列表

# 保存Dataset（会自动生成arrow文件和元数据）
dataset.save_to_disk('dataset/SciERC/train_data/arrow_files/RM/train')