import json
import os
import numpy as np


class SimpleTokenizer:
    def __init__(self, chars=None):
        if chars is not None:
            self.chars = sorted(list(set(chars)))
            self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
            self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        else:
            self.chars = []
            self.char_to_id = {}
            self.id_to_char = {}

    @property
    def vocab_size(self):
        return len(self.chars)

    def fit(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_id.get(ch, 0) for ch in text]

    def decode(self, token_ids):
        return "".join([self.id_to_char.get(int(i), "") for i in token_ids])

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "chars": self.chars,
            "char_to_id": self.char_to_id,
            "id_to_char": {str(k): v for k, v in self.id_to_char.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.chars = data["chars"]
        tokenizer.char_to_id = data["char_to_id"]
        tokenizer.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        return tokenizer


def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def create_sequences(text, tokenizer, seq_length=32, step=1):
    tokens = tokenizer.encode(text)
    inputs = []
    targets = []
    for i in range(0, len(tokens) - seq_length, step):
        inputs.append(tokens[i : i + seq_length])
        targets.append(tokens[i + 1 : i + seq_length + 1])
    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_file = os.path.join(current_dir, "data", "sample.txt")
    raw_text = load_data(sample_file)

    tokenizer = SimpleTokenizer()
    tokenizer.fit(raw_text)

    print(f"Dataset length: {len(raw_text)} characters")
    print(f"Vocabulary size: {tokenizer.vocab_size} unique characters")

    X, y = create_sequences(raw_text, tokenizer, seq_length=32, step=3)
    print(f"Inputs shape: {X.shape}")
    print(f"Targets shape: {y.shape}")

    sample_encoded = tokenizer.encode("Hello")
    sample_decoded = tokenizer.decode(sample_encoded)
    print(f"Sample encode: 'Hello' -> {sample_encoded}")
    print(f"Sample decode: {sample_encoded} -> '{sample_decoded}'")
