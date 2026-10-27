import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from data_loader import SimpleTokenizer
from model import build_model


def load_model_and_tokenizer(save_dir="saved_model",
                             seq_length=32,
                             d_model=64,
                             d_ff=128,
                             num_layers=2):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_save_dir = os.path.join(current_dir, save_dir) if not os.path.isabs(save_dir) else save_dir

    vocab_path = os.path.join(resolved_save_dir, "vocab.json")
    weights_path = os.path.join(resolved_save_dir, "model.weights.h5")

    if not os.path.exists(vocab_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model files not found in {resolved_save_dir}. Please run train.py first."
        )

    tokenizer = SimpleTokenizer.load(vocab_path)
    model = build_model(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_length,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers
    )
    model.load_weights(weights_path)
    return model, tokenizer


def predict_next_token(model, tokenizer, prompt, temperature=1.0, top_k=5):
    seq_length = model.max_seq_len
    tokens = tokenizer.encode(prompt)

    if not tokens:
        tokens = [0]

    # Crop to max_seq_len
    if len(tokens) > seq_length:
        tokens = tokens[-seq_length:]

    input_tokens = np.array([tokens], dtype=np.int32)
    logits = model(input_tokens)
    last_token_logits = logits[0, -1, :].numpy()

    # Temperature scaling
    if temperature <= 0.01:
        next_id = int(np.argmax(last_token_logits))
    else:
        scaled_logits = last_token_logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probabilities = exp_logits / np.sum(exp_logits)
        next_id = int(np.random.choice(len(probabilities), p=probabilities))

    # Top-k candidates for inspection
    top_indices = np.argsort(last_token_logits)[-top_k:][::-1]
    top_candidates = [(tokenizer.decode([idx]), float(last_token_logits[idx])) for idx in top_indices]

    next_char = tokenizer.decode([next_id])
    return next_char, top_candidates


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7):
    generated = prompt
    for _ in range(max_new_tokens):
        next_char, _ = predict_next_token(model, tokenizer, generated, temperature=temperature)
        generated += next_char
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Next-Token Prediction and Generation")
    parser.add_argument("--prompt", type=str, default="The Fox and the", help="Initial text prompt")
    parser.add_argument("--tokens", type=int, default=120, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (lower = more deterministic)")
    args = parser.parse_args()

    print("=" * 60)
    print("Transformer Text Generation")
    print("=" * 60)

    try:
        model, tokenizer = load_model_and_tokenizer()
    except FileNotFoundError as e:
        print(e)
        exit(1)

    print(f"Prompt: \"{args.prompt}\"")
    next_char, candidates = predict_next_token(model, tokenizer, args.prompt, temperature=0.0)
    print(f"\nImmediate next token prediction (greedy): '{next_char}'")
    print("Top 5 candidate tokens and logits:")
    for char, score in candidates:
        display_char = repr(char)
        print(f"  Token: {display_char:<6} Logit: {score:.3f}")

    print("\nGenerating text continuation...")
    result = generate_text(model, tokenizer, args.prompt, max_new_tokens=args.tokens, temperature=args.temperature)
    print("\n--- Output ---")
    print(result)
    print("--------------")
