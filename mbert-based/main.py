import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from configs import SAVE_PATH, GOLD_TRAIN_PATH, GOLD_DEV_PATH

# Ensure CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the dataset
def load_jsonl_dataset(file_path):
    return load_dataset('json', data_files=file_path)

# Paths to your datasets
train_path = GOLD_TRAIN_PATH
val_path = GOLD_DEV_PATH

# Load and prepare datasets
dataset = load_dataset('json', data_files=train_path)
dataset = dataset['train'].train_test_split(test_size=0.2)

# Check if 'label' needs to be generated from 'languages'
if 'languages' in dataset['train'].column_names:
    languages = np.unique(dataset['train']['languages'])
    lang2id = {lang: idx for idx, lang in enumerate(languages)}

    # Function to map languages to ids
    def add_label_column(examples):
        examples['label'] = [lang2id[lang] for lang in examples['languages']]
        return examples

    # Apply the function to map languages to labels
    dataset = dataset.map(add_label_column)

# Initialize the tokenizer for Multilingual BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# Apply the tokenization function
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# DataLoader
batch_size = 16
train_loader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(tokenized_datasets['test'], batch_size=batch_size)

# Initialize the model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(lang2id))
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

# Save the fine-tuned model and tokenizer
model_path = SAVE_PATH
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)