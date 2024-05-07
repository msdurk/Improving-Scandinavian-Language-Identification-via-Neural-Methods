from transformers import AutoTokenizer
from datasets import Dataset
import re

def tokenize_dataset(dataset, model_checkpoint='facebook/nllb-200-distilled-600M', max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    tokenized_dataset = dataset.map(
        lambda examples: {'input_ids': tokenizer(clean_text(examples['text']), truncation=True, padding="max_length", max_length=max_length)['input_ids']},
        batched=True
    )
    
    return tokenized_dataset

def clean_text(texts):
    cleaned_texts = []
    for text in texts:
        cleaned = text.lower()
        cleaned = re.sub(r'\W', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned_texts.append(cleaned)
    return cleaned_texts

