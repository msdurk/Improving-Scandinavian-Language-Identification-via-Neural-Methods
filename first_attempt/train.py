from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

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
    
def run_trainer(model, train_dataset, eval_dataset):
    trainer = Trainer(
        model=model,
        args=setup_training_arguments(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset 
    )
    trainer.train()
    model.save_pretrained('./saved_model')  
