from transformers import BertTokenizer
from configs import MBERT_PATH
import re
import torch

def clean_text(text):
    cleaned = text.lower()
    cleaned = re.sub(r'\W', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def tokenize_dataset(dataset, tokenizer_path=MBERT_PATH, max_length=128):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    tokenized_dataset = []
    for example in dataset:
        text = example['text']
        cleaned_text = clean_text(text)
        tokenized_text = tokenizer(cleaned_text, truncation=True, padding="max_length", max_length=max_length)

        # Create a label vector directly from label_ids
        label_vector = [0] * 5  # Assuming there are 5 possible labels
        label_vector[example['label_ids']] = 1

        tokenized_example = {
            'input_ids': tokenized_text['input_ids'],
            'attention_mask': tokenized_text['attention_mask'],
            'labels': label_vector
        }
        tokenized_dataset.append(tokenized_example)

    return tokenized_dataset
