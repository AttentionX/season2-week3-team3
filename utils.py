from typing import List
import json

def load_from_jsonl(path: str) -> List[str]:
    result = []
    with open(path, 'r', encoding='utf-8') as file:
        result = json.load(file)
    return result


def save_json_file(file_path, data):
    json_string = json.dumps(data, indent=4)
    with open(file_path, "w") as json_file:
        json_file.write(json_string)