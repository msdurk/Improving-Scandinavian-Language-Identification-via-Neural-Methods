from transformers import Trainer, TrainingArguments
import torch
import fasttext
from transformers import AutoModelForSequenceClassification
from custom_model import CustomModel
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from data import load_and_filter_data
from first_attempt.preprocess import tokenize_dataset
from torch.utils.data import DataLoader
from configs import SAVE_PATH, GOLD_DEV_PATH, GOLD_TRAIN_PATH, MBERT_PATH
from transformers import DataCollatorWithPadding, BertTokenizer


def save_model(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def main():
    train_filepath = GOLD_TRAIN_PATH
    dev_filepath = GOLD_DEV_PATH

    # Load fastText model
    fasttext_model_path = hf_hub_download(repo_id="laurievb/OpenLID", filename="model.bin")
    fasttext_model = fasttext.load_model(fasttext_model_path)

    mbert_model_path = MBERT_PATH
    mbert_model = AutoModelForSequenceClassification.from_pretrained(mbert_model_path, num_labels=5)

    tokenizer = BertTokenizer.from_pretrained(MBERT_PATH)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    custom_model = CustomModel(fasttext_model, mbert_model)

    train_data = load_and_filter_data(train_filepath)
    train_data = tokenize_dataset(train_data)
    dev_data = load_and_filter_data(dev_filepath)
    dev_data = tokenize_dataset(dev_data)

    def setup_training_arguments():
        return TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",  
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            save_strategy="epoch" 
        )
    training_args = setup_training_arguments()

    trainer = Trainer(
        model=custom_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data
    )
    trainer.train()

    results = trainer.evaluate(eval_dataset=dev_data)
    print(results)
    save_path = SAVE_PATH
    save_model(custom_model, tokenizer, save_path)

if __name__ == "__main__":
    main()
