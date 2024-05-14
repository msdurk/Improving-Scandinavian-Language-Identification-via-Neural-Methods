import time
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from configs import SAVE_DIR, save_count, GOLD_TRAIN_PATH, GOLD_TRAIN_SHORT_PATH, GOLD_DEV_PATH, GOLD_DEV_SHORT_PATH, SILVER_TRAIN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def load_jsonl_dataset(file_path):
    return load_dataset('json', data_files=file_path)

train_path = SILVER_TRAIN
val_path = GOLD_DEV_SHORT_PATH
MODEL_NAME = 'bert-base-multilingual-cased'
batch_size = 16
# Initialize the tokenizer for Multilingual BERT
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

def add_language2id(dataset):
   # Check if 'label' needs to be generated from 'languages'
    if 'languages' in dataset['train'].column_names:
        languages = np.unique(dataset['train']['languages'])
        lang2id = {lang: idx for idx, lang in enumerate(languages)}

        # Function to map languages to ids
        def add_label_column(examples):
            examples['label'] = [lang2id[lang] for lang in examples['languages']]
            return examples

        # Apply the function to map languages to labels
        return dataset.map(add_label_column), lang2id
    return dataset, lang2id

if train_path == SILVER_TRAIN:
    dataset = load_dataset('json', data_files=train_path)
    dataset = dataset['train'].train_test_split(test_size=0.2)
    dataset, lang2id = add_language2id(dataset)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    train_loader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_datasets['test'], batch_size=batch_size)
else:
    train_dataset = load_dataset('json', data_files=train_path)
    val_dataset = load_dataset('json', data_files=val_path)

    train_dataset, lang2id = add_language2id(train_dataset)
    val_dataset, _ = add_language2id(val_dataset)

    tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
    tokenized_train_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    tokenized_val_datasets = val_dataset.map(tokenize_function, batched=True)
    tokenized_val_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    train_loader = DataLoader(tokenized_train_datasets['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_val_datasets['train'], batch_size=batch_size)

# Initialize the model for sequence classification
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(lang2id))
model.to(device)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)

# Number of training epochs
epochs = 3

# Total number of training steps is the number of batches * number of epochs
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Function to calculate accuracy
def compute_accuracy(preds, labels):
    return accuracy_score(labels.cpu(), preds.argmax(dim=-1).cpu())

# Training loop
start = time.time()
for epoch in range(epochs):
    model.train()
    total_loss, total_accuracy = 0, 0

    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('label')

        # Forward pass, calculate loss, backpropagation, and update model weights
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Accumulate the loss and accuracy
        total_loss += loss.item()
        total_accuracy += compute_accuracy(outputs.logits, labels)

    # Calculate average loss and accuracy for the training epoch
    avg_train_loss = total_loss / len(train_loader)
    avg_train_accuracy = total_accuracy / len(train_loader)
    print(f"Average training loss: {avg_train_loss}")
    print(f"Average training accuracy: {avg_train_accuracy}")

    # Evaluation loop
    model.eval()
    total_val_accuracy, total_val_loss = 0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}'):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('label')

            # Forward pass and calculate loss for validation
            outputs = model(**batch, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()
            total_val_accuracy += compute_accuracy(outputs.logits, labels)

    # Calculate average loss and accuracy for the validation epoch
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_accuracy = total_val_accuracy / len(val_loader)
    print(f"Average validation loss: {avg_val_loss}")
    print(f"Average validation accuracy: {avg_val_accuracy}")

end = time.time() - start
print(f"Total training time for model {MODEL_NAME} trained on {train_path} validated on {val_path} is {end} seconds.")

# Save the fine-tuned model and tokenizer
model_path = SAVE_DIR + "/" + MODEL_NAME + "_"
model_path += train_path.split('/')[-1].split('.')[0] + "_"
model_path += val_path.split('/')[-1].split('.')[0]
model.config.lang2id = lang2id
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"Saved model to {model_path}")