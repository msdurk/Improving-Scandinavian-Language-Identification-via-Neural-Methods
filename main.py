from transformers import Trainer, TrainingArguments
import torch
import fasttext
from transformers import AutoModelForSequenceClassification
from custom_model import CustomModel
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from data import load_and_filter_data
from preprocess import tokenize_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, BertTokenizer

def main():
    train_filepath = '/fp/homes01/u01/ec-victocla/exam/gold_train.jsonl'
    dev_filepath = '/fp/homes01/u01/ec-victocla/exam/gold_dev.jsonl'

    # Load fastText model
    fasttext_model_path = hf_hub_download(repo_id="laurievb/OpenLID", filename="model.bin")
    fasttext_model = fasttext.load_model(fasttext_model_path)

    mbert_model_path = '/fp/projects01/ec30/models/bert-base-multilingual-cased/'
    mbert_model = AutoModelForSequenceClassification.from_pretrained(mbert_model_path, num_labels=5)

    tokenizer = BertTokenizer.from_pretrained('/fp/projects01/ec30/models/bert-base-multilingual-cased/')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    custom_model = CustomModel(fasttext_model, mbert_model)

    train_data = load_and_filter_data(train_filepath)
    train_data = tokenize_dataset(train_data)

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

    # Train the model
    trainer = Trainer(
        model=custom_model,
        args=training_args,
        train_dataset=train_data
    )
    trainer.train()

    dev_data = load_and_filter_data(dev_filepath)
    dev_data = tokenize_dataset(dev_data)
    results = trainer.evaluate(eval_dataset=dev_data)
    print(results)

if __name__ == "__main__":
    main()
