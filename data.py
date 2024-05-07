import json
from datasets import Dataset

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def filter_data(dataset, max_length=6):
    filtered_data = [item for item in dataset if item.get('text') and len(item['text'].split()) <= max_length]

    if not filtered_data:
        return []

    for item in filtered_data:
        if 'languages' in item:
            language_label = item['languages'][0]  
            label_ids = map_language_to_label_id(language_label)
            item['label_ids'] = label_ids
            item['label'] = language_label
    return filtered_data

def map_language_to_label_id(language_label):
    label_id_mapping = {'nn': 0, 'nb': 1, 'da': 2, 'sv': 3, 'other': 4}
    return label_id_mapping.get(language_label, 4)  

def load_and_filter_data(filepath):
    data = load_data(filepath)
    filtered_data = filter_data(data)
    for item in filtered_data:
        item['label'] = item['languages'][0]
        item['label_ids'] = map_language_to_label_id(item['label'])
        #item.pop('languages', None)  
    print(filtered_data[1])
    return filtered_data
