import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras


class CausalSelfAttention(keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.q_dense = keras.layers.Dense(d_model)
        self.k_dense = keras.layers.Dense(d_model)
        self.v_dense = keras.layers.Dense(d_model)
        self.out_dense = keras.layers.Dense(d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        
        Q = self.q_dense(x)
        K = self.k_dense(x)
        V = self.v_dense(x)

        # Scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Causal mask: lower-triangular matrix prevents attending to future tokens
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = tf.expand_dims(mask, 0)

        # Apply large negative value to future positions
        scores = tf.where(tf.equal(mask, 1.0), scores, -1e9)
        weights = tf.nn.softmax(scores, axis=-1)

        context = tf.matmul(weights, V)
        return self.out_dense(context)


class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model, d_ff, **kwargs):
        super().__init__(**kwargs)
        self.attention = CausalSelfAttention(d_model)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.ffn = keras.Sequential([
            keras.layers.Dense(d_ff, activation="relu"),
            keras.layers.Dense(d_model)
        ])

    def call(self, x):
        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x)
        x = x + attn_out

        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        return x


class TransformerLanguageModel(keras.Model):
    def __init__(self, vocab_size, max_seq_len, d_model=64, d_ff=128, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers

        self.token_embedding = keras.layers.Embedding(vocab_size, d_model)
        self.pos_embedding = keras.layers.Embedding(max_seq_len, d_model)
        self.blocks = [TransformerBlock(d_model, d_ff) for _ in range(num_layers)]
        self.final_norm = keras.layers.LayerNormalization(epsilon=1e-5)
        self.head = keras.layers.Dense(vocab_size)

    def call(self, token_ids):
        seq_len = tf.shape(token_ids)[1]
        positions = tf.range(0, seq_len, delta=1)
        positions = tf.expand_dims(positions, 0)

        x = self.token_embedding(token_ids) + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.head(x)
        return logits


def build_model(vocab_size, max_seq_len=32, d_model=64, d_ff=128, num_layers=2):
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers
    )
    dummy_input = tf.zeros((1, max_seq_len), dtype=tf.int32)
    model(dummy_input)
    return model


if __name__ == "__main__":
    test_vocab_size = 50
    test_seq_len = 32
    model = build_model(vocab_size=test_vocab_size, max_seq_len=test_seq_len)
    model.summary()

    sample_input = tf.random.uniform((2, test_seq_len), minval=0, maxval=test_vocab_size, dtype=tf.int32)
    sample_output = model(sample_input)
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Model output shape (logits): {sample_output.shape}")
