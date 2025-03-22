import torch
from transformers import AutoModelForSequenceClassification
from config import MODEL_PATH


class BertModel:
    def __init__(self, model_name="bert-base-multilingual-cased", num_labels=5):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.nn.functional.softmax(outputs.logits, dim=-1)

    def save(self):
        self.model.save_pretrained(MODEL_PATH)

    def load(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Использование:
# model = BertModel()
# model.save()
# predictions = model.predict(tokenizer.tokenize("Как настроить API?"))
# print(predictions)
