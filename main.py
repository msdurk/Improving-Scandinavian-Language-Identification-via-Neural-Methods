from preprocess import tokenize_dataset
from train import run_trainer
from data import load_and_filter_data
from custom_model import CustomModel
from huggingface_hub import hf_hub_download
import fasttext
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification


def main():
    train_filepath = '/fp/homes01/u01/ec-victocla/exam/gold_train.jsonl'
    dev_filepath = '/fp/homes01/u01/ec-victocla/exam/gold_dev.jsonl'

    fasttext_model_path = hf_hub_download(repo_id="laurievb/OpenLID", filename="model.bin")
    fasttext_model = fasttext.load_model(fasttext_model_path)

    mbert_model_path = '/fp/projects01/ec30/models/bert-base-multilingual-cased/'
    mbert_model = AutoModelForSequenceClassification.from_pretrained(mbert_model_path)

    custom_model = CustomModel(fasttext_model, mbert_model)

    train_data = load_and_filter_data(train_filepath)
    dev_data = load_and_filter_data(dev_filepath)

    train_data = tokenize_dataset(train_data)
    dev_data = tokenize_dataset(dev_data)

    num_labels = 5
    

    run_trainer(custom_model, train_data, dev_data)

if __name__ == "__main__":
    main()
