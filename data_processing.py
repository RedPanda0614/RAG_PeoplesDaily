import json
import os
from tqdm import tqdm


def clean_json_files(directory):
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # 保留不符合条件的条目
            cleaned_data = [
                item for item in data if '本版责编' not in item['title'] and item['content'].strip() != '']

            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(cleaned_data, file, ensure_ascii=False, indent=4)


# 指定存放JSON文件的文件夹路径
directory = 'news'

clean_json_files(directory)
