import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from data_loader import SimpleTokenizer, create_sequences, load_data
from model import build_model
from predict import generate_text, load_model_and_tokenizer, predict_next_token
from train import train


def run_tests():
    print("=" * 60)
    print("Running Project Verification Tests")
    print("=" * 60)

    # 1. Test dataset loading
    sample_file = os.path.join(os.path.dirname(__file__), "data", "sample.txt")
    assert os.path.exists(sample_file), f"Dataset not found at {sample_file}"
    raw_text = load_data(sample_file)
    assert len(raw_text) > 0, "Dataset text is empty"
    print("[PASS] 1. Dataset loading passed.")

    # 2. Test tokenization
    tokenizer = SimpleTokenizer()
    tokenizer.fit(raw_text)
    assert tokenizer.vocab_size > 0, "Vocabulary size is 0"
    test_phrase = "Transformer Next Token"
    encoded = tokenizer.encode(test_phrase)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_phrase, f"Tokenization mismatch: '{decoded}' != '{test_phrase}'"
    print(f"[PASS] 2. Tokenization and vocabulary passed (Vocab size: {tokenizer.vocab_size}).")

    # 3. Test sequence shapes
    seq_len = 16
    X, y = create_sequences(raw_text, tokenizer, seq_length=seq_len, step=4)
    assert X.shape[1] == seq_len, f"Expected input sequence length {seq_len}, got {X.shape[1]}"
    assert y.shape[1] == seq_len, f"Expected target sequence length {seq_len}, got {y.shape[1]}"
    assert X.shape[0] == y.shape[0], "Input and target sample counts do not match"
    print(f"[PASS] 3. Sequence preparation passed (X shape: {X.shape}, y shape: {y.shape}).")

    # 4. Test model architecture and causal masking
    model = build_model(vocab_size=tokenizer.vocab_size, max_seq_len=seq_len, d_model=32, d_ff=64, num_layers=1)
    
    # Prefix invariance test: changing tokens after prefix should not affect prefix logits
    prefix_len = 6
    seq_a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    seq_b = [1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    in_a = tf.constant([seq_a], dtype=tf.int32)
    in_b = tf.constant([seq_b], dtype=tf.int32)

    logits_a = model(in_a).numpy()
    logits_b = model(in_b).numpy()

    max_prefix_diff = np.max(np.abs(logits_a[0, :prefix_len, :] - logits_b[0, :prefix_len, :]))
    assert max_prefix_diff < 1e-4, f"Causal attention leak! Max diff: {max_prefix_diff}"
    print(f"[PASS] 4. Causal self-attention verification passed (Max prefix difference: {max_prefix_diff:.2e}).")

    # 5. Test model saving and loading
    test_save_dir = os.path.join(os.path.dirname(__file__), "saved_model_test")
    os.makedirs(test_save_dir, exist_ok=True)
    test_weights = os.path.join(test_save_dir, "model.weights.h5")
    test_vocab = os.path.join(test_save_dir, "vocab.json")

    model.save_weights(test_weights)
    tokenizer.save(test_vocab)

    loaded_tok = SimpleTokenizer.load(test_vocab)
    loaded_model = build_model(vocab_size=loaded_tok.vocab_size, max_seq_len=seq_len, d_model=32, d_ff=64, num_layers=1)
    loaded_model.load_weights(test_weights)

    out_orig = model(in_a).numpy()
    out_loaded = loaded_model(in_a).numpy()
    max_load_diff = np.max(np.abs(out_orig - out_loaded))
    assert max_load_diff < 1e-5, f"Loaded weights mismatch! Max diff: {max_load_diff}"
    print(f"[PASS] 5. Model save and reload verification passed (Max diff: {max_load_diff:.2e}).")

    # Clean up test save directory
    try:
        os.remove(test_weights)
        os.remove(test_vocab)
        os.rmdir(test_save_dir)
    except Exception:
        pass

    print("\nAll verification tests passed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Transformer Next-Token Prediction System")
    parser.add_argument("--mode", type=str, choices=["all", "train", "generate", "test"], default="all",
                        help="Execution mode: all (train+test+generate), train, generate, or test")
    parser.add_argument("--prompt", type=str, default="The Tortoise", help="Prompt for generation")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--tokens", type=int, default=100, help="Tokens to generate")
    args = parser.parse_args()

    if args.mode == "test":
        run_tests()
    elif args.mode == "train":
        train(epochs=args.epochs)
    elif args.mode == "generate":
        model, tokenizer = load_model_and_tokenizer()
        result = generate_text(model, tokenizer, args.prompt, max_new_tokens=args.tokens)
        print("\nGenerated output:")
        print(result)
    else:
        # Default 'all' pipeline
        print("Running full pipeline: Verification -> Training -> Generation")
        run_tests()
        print("\n" + "=" * 60)
        history = train(epochs=args.epochs)
        print("\n" + "=" * 60)
        print("Loading trained model for generation demonstration...")
        model, tokenizer = load_model_and_tokenizer()

        sample_prompts = ["The Tortoise", "A hungry Fox", "The Lion"]
        for p in sample_prompts:
            print(f"\nPrompt: \"{p}\"")
            next_ch, _ = predict_next_token(model, tokenizer, p, temperature=0.0)
            print(f"Greedy next-token prediction: '{next_ch}'")
            sample_gen = generate_text(model, tokenizer, p, max_new_tokens=60, temperature=0.7)
            print(f"Generated text: {sample_gen}")


if __name__ == "__main__":
    main()
