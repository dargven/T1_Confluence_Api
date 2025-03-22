from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from config import MODEL_PATH, TOKENIZER_PATH, TRAIN_CSV, VAL_CSV

class ModelTrainer:
    def __init__(self, model_name="bert-base-multilingual-cased", num_labels=5):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def load_data(self):
        dataset = load_dataset("csv", data_files={"train": TRAIN_CSV, "validation": VAL_CSV})
        return dataset

    def tokenize_data(self, dataset):
        return dataset.map(lambda examples: self.tokenizer(examples["text"], padding="max_length", truncation=True), batched=True)

    def train(self):
        dataset = self.load_data()
        dataset = self.tokenize_data(dataset)

        training_args = TrainingArguments(
            output_dir=MODEL_PATH,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
        )

        trainer.train()
        self.model.save_pretrained(MODEL_PATH)
        self.tokenizer.save_pretrained(TOKENIZER_PATH)

# Использование:
# trainer = ModelTrainer()
# trainer.train()
