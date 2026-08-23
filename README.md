# LLM From Scratch - Learning Journey 🚀

A documentation and code repository tracking my progress through the book ***Build a Large Language Model (From Scratch)*** by Sebastian Raschka. This project implements the core components of a generative language model step-by-step from scratch using Python and PyTorch.

---

## 🗺️ Current Progress: Chapter 3

*   **Chapter 1:** Understanding Large Language Models (Concepts & Overview)
*   **Chapter 2:** Working with Text Data (Tokenization, BPE, and PyTorch Datasets)
*   **Chapter 3:** Coding Attention Mechanisms (Self-Attention, Causal Attention, and Multi-Head Attention) 📍 *(Current Focus)*

---

## 📚 Key Concepts Learned & Implemented

### 1. The Data Pipeline (Chapter 2)
*   **Tokenization & Vocabulary Construction:** Converting raw text into individual tokens and mapping them to integer IDs using a custom or pre-built vocabulary.
*   **Byte-Pair Encoding (BPE):** Understanding how subword tokenization handles out-of-vocabulary words (similar to GPT models using Tiktoken).
*   **Sliding Window Data Loaders:** Creating input-target pairs (`x` and `y`) using PyTorch's `Dataset` and `DataLoader` to train the model on next-token prediction.
*   **Token Embeddings:** Mapping integer token IDs into dense continuous vector representations ($\mathbb{R}^{d}$) alongside absolute positional embeddings.

### 2. Attention Mechanisms (Chapter 3)
*   **Self-Attention Intuition:** Moving beyond standard RNNs and feed-forward networks by allowing tokens to dynamically focus on relevant parts of the input sequence.
*   **Scaled Dot-Product Attention:** 
    *   Computing **Queries ($Q$)**, **Keys ($K$)**, and **Values ($V$)** via linear weight projections.
    *   Calculating attention weights using matrix multiplication and scaling by $\sqrt{d_k}$ to stabilize gradients.
    *   Applying the **Softmax** function to obtain normalized attention probabilities.
*   **Causal Attention (Masked Attention):** 
    *   Implementing future-token masking (setting upper-triangular weights to $-\infty$) so the model cannot "cheat" by looking ahead during autoregressive generation.
*   **Multi-Head Attention:** 
    *   Splitting query, key, and value tensors into multiple "heads" to allow the model to jointly attend to information from different representation subspaces simultaneously.
    *   Implementing an efficient modular wrapper (`MultiHeadAttention`) combining multiple causal attention blocks.

---

## 🛠️ Code Structure

```text
├── chapter02/
│   ├── text_preprocessing.py   # Tokenization and vocabulary mapping
│   └── data_loader.py          # PyTorch Dataset and batching logic
├── chapter03/
│   ├── self_attention.py       # Simplified and dot-product self-attention
│   ├── causal_attention.py     # Masked attention for autoregressive models
│   └── multi_head_attention.py # Multi-head attention wrapper
└── README.md