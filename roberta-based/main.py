import torch
from datasets import load_dataset, Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from configs import SAVE_PATH, GOLD_TRAIN_PATH, GOLD_DEV_PATH
import numpy as np
from sklearn.metrics import accuracy_score

# Step 1: Load your dataset
def load_jsonl_dataset(file_path):
    return load_dataset('json', data_files={'train': file_path})

dataset = load_jsonl_dataset(GOLD_TRAIN_PATH)

# Step 2: Tokenize the data
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 3: Define the model
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2)

# Step 4: Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # perform evaluation at the end of each epoch
)

# Step 5: Define the compute_metrics function for evaluation
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

# Ensure all data are in the same format (padding and converting to tensors)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Step 6: Initialize the Trainer
trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_datasets['train'],  # training dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    data_collator=data_collator,
)

# Step 7: Train the model
trainer.train()

# Step 8: Save the fine-tuned model
model_path = SAVE_PATH
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
