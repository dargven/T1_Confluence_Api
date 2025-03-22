from transformers import AutoTokenizer
from config import TOKENIZER_PATH

class Tokenizer:
    def __init__(self, model_name="bert-base-multilingual-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text):
        return self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")

    def save(self):
        self.tokenizer.save_pretrained(TOKENIZER_PATH)

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Использование:
# tokenizer = Tokenizer()
# tokenizer.save()
# tokens = tokenizer.tokenize("Пример текста для токенизации.")
# print(tokens)
