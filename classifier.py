from transformers import pipeline
from config import MODEL_PATH

class DocumentClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.classifier = pipeline("text-classification", model=model_path)

    def classify(self, text):
        return self.classifier(text)

# Использование:
# classifier = DocumentClassifier()
# result = classifier.classify("Описание API для интеграции с системой X")
# print(result)
