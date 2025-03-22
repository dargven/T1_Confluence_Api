import json
import pandas as pd
from config import DATA_PATH, TRAIN_CSV, VAL_CSV
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path

    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def prepare_dataset(self, df):
        df["label"] = df["project"].astype("category").cat.codes
        train_texts, val_texts, train_labels, val_labels = train_test_split(df["content"], df["label"], test_size=0.2)

        train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
        val_df = pd.DataFrame({"text": val_texts, "label": val_labels})

        train_df.to_csv(TRAIN_CSV, index=False)
        val_df.to_csv(VAL_CSV, index=False)

    def process(self):
        df = self.load_data()
        self.prepare_dataset(df)

# Использование:
# processor = DataProcessor()
# processor.process()
