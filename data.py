import json
from datasets import load_dataset, Dataset, DatasetDict

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data


def filter_data(dataset, max_length=6):
    filtered_data = [item for item in dataset if item['text'] and len(item['text'].split()) <= max_length]

    if not filtered_data:
        return Dataset.from_dict({})

    data_dict = {}
    for key in filtered_data[0].keys():
        data_dict[key] = [item[key] for item in filtered_data]

    dataset = Dataset.from_dict(data_dict)
    return dataset


def load_and_filter_data(filepath):
    data = load_data(filepath)
    filtered_data = filter_data(data)
    return filtered_data
    