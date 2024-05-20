from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments


dataset = load_dataset('json', data_files={'train': 'gold_train.jsonl', 'test': 'gold_dev.jsonl'})


def filter_criteria(example):
    if len(example['text'].split()) > 6:
        return False
    if '«' in example['text'] or '»' in example['text']:
        return False
    return True


#filtered_dataset = {split: ds.filter(filter_criteria) for split, ds in dataset.items()}
filtered_dataset = dataset


label_names = ['nn', 'nb', 'sv', 'dn', 'other']
label_dict = {label: i for i, label in enumerate(label_names)}

def label_to_id(example):
    language = example['languages'][0] if example['languages'] else 'other'
    example['label'] = label_dict.get(language, label_dict['other'])
    return example

for split in filtered_dataset:
    filtered_dataset[split] = filtered_dataset[split].map(label_to_id, remove_columns=['languages'])



tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=len(label_names))


def tokenize_data(example):
    return tokenizer(example['text'], padding="max_length", truncation=True, max_length=128)


tokenized_data = {split: ds.map(tokenize_data, batched=True) for split, ds in filtered_dataset.items()}


training_args = TrainingArguments(
    output_dir='./lablelable',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    warmup_steps=500,
    save_steps=1000000, 
    save_total_limit=1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100000,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test']
)


trainer.train()


evaluation_results = trainer.evaluate()
print(evaluation_results)

model_save_path = '/fp/projects01/ec30/mathisdu/gold_long/e5'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)


