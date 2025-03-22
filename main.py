from confluence_api import ConfluenceAPI
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from classifier import DocumentClassifier


def main():
    api = ConfluenceAPI()
    documents = api.get_pages("DEV_DOCS")

    # Сохраняем в файл
    import json
    with open("data/documents.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)

    processor = DataProcessor()
    processor.process()

    trainer = ModelTrainer(num_labels=5)  # Количество проектов
    trainer.train()

    classifier = DocumentClassifier()
    sample_text = "Описание API для интеграции с системой X"
    prediction = classifier.classify(sample_text)
    print(f" Предсказание: {prediction}")


if __name__ == "__main__":
    main()
