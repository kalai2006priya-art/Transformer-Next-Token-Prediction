import argparse
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from data_loader import SimpleTokenizer, create_sequences, load_data
from model import build_model


def train(data_path="data/sample.txt",
          save_dir="saved_model",
          epochs=20,
          batch_size=32,
          seq_length=32,
          d_model=64,
          d_ff=128,
          num_layers=2,
          learning_rate=0.003):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_data_path = os.path.join(current_dir, data_path) if not os.path.isabs(data_path) else data_path
    resolved_save_dir = os.path.join(current_dir, save_dir) if not os.path.isabs(save_dir) else save_dir

    print("=" * 60)
    print("Training Transformer Next-Token Prediction Model")
    print("=" * 60)
    print(f"Dataset path: {resolved_data_path}")

    raw_text = load_data(resolved_data_path)
    tokenizer = SimpleTokenizer()
    tokenizer.fit(raw_text)

    vocab_size = tokenizer.vocab_size
    print(f"Dataset size: {len(raw_text)} characters")
    print(f"Vocabulary size: {vocab_size} unique tokens")

    X, y = create_sequences(raw_text, tokenizer, seq_length=seq_length, step=2)
    print(f"Training sequences: {X.shape[0]} samples (seq_length = {seq_length})")

    model = build_model(
        vocab_size=vocab_size,
        max_seq_len=seq_length,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    print("\nStarting training...")
    start_time = time.time()
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds.")

    os.makedirs(resolved_save_dir, exist_ok=True)
    weights_path = os.path.join(resolved_save_dir, "model.weights.h5")
    vocab_path = os.path.join(resolved_save_dir, "vocab.json")

    model.save_weights(weights_path)
    tokenizer.save(vocab_path)

    initial_loss = history.history["loss"][0]
    final_loss = history.history["loss"][-1]
    initial_acc = history.history["accuracy"][0]
    final_acc = history.history["accuracy"][-1]

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Initial Loss: {initial_loss:.4f} | Initial Accuracy: {initial_acc * 100:.2f}%")
    print(f"Final Loss:   {final_loss:.4f} | Final Accuracy:   {final_acc * 100:.2f}%")
    print(f"Model weights saved to: {weights_path}")
    print(f"Vocabulary saved to:    {vocab_path}")

    return history.history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Next-Token Prediction Transformer")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=32, help="Sequence length")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        learning_rate=args.lr
    )
