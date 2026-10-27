# Development of a Transformer-Based Next-Token Prediction Model Using Basic Self-Attention

An educational, lightweight Transformer language model built from scratch using Python and TensorFlow/Keras to demonstrate causal self-attention and next-token prediction.

---

## Architecture Overview

1. **Token & Positional Embeddings**: Maps character token IDs and positions into dense vectors ($d_{model} = 64$).
2. **Causal Self-Attention**: Computes scaled dot-product attention ($Q K^T / \sqrt{d_k}$) with a lower-triangular causal attention mask that prevents attending to future positions.
3. **Transformer Block**: Combines causal self-attention, layer normalization, residual connections, and a feed-forward network ($d_{ff} = 128$).
4. **Output Head**: Projects final normalized states to vocabulary logits for next-token probability distribution.

---

## Project Structure

```text
Transformer_Next_Token_Prediction/
├── data/
│   └── sample.txt        # Public-domain sample text dataset
├── data_loader.py        # Tokenizer, vocabulary builder, and sequence generator
├── model.py              # Causal self-attention layer and Transformer architecture
├── train.py              # Model training loop, loss tracking, and checkpoint saving
├── predict.py            # Next-token prediction and autoregressive text generation
├── main.py               # Unified CLI runner and verification test suite
├── requirements.txt      # Dependency specification
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation
```

---

## Setup Instructions

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Install Dependencies
Open a PowerShell terminal in the project directory and install the required packages:

```powershell
pip install -r requirements.txt
```

---

## Usage

### 1. Run Verification Tests
Verifies dataset loading, tokenization, sequence shapes, causal mask invariance, and model weight reload:

```powershell
python main.py --mode test
```

### 2. Train the Model
Trains the Transformer on `data/sample.txt` and saves weights to `saved_model/model.weights.h5`:

```powershell
python train.py --epochs 25 --batch_size 32
```

### 3. Generate Text / Predict Next Token
Predict the immediate next token and generate continuation text from a prompt:

```powershell
python predict.py --prompt "The Tortoise and the" --tokens 100 --temperature 0.6
```

### 4. Run the Complete Pipeline
Runs verification tests, trains the model, and displays generation samples in a single command:

```powershell
python main.py
```

---

## Real Training Metrics (Recorded)

- **Dataset Size**: 3,499 characters
- **Vocabulary Size**: 49 unique character tokens
- **Training Samples**: 1,734 sequences (sequence length = 32)
- **Model Parameters**: 75,570
- **Epochs**: 25
- **Training Duration**: ~31.7 seconds (CPU)
- **Initial Loss**: 2.7425 | **Initial Accuracy**: 26.20%
- **Final Loss**: 0.2918 | **Final Accuracy**: 90.79%
