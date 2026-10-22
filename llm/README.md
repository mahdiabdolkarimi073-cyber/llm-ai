# Minimal AI: From-Scratch GPT-Style LLM

A complete decoder-only Transformer language model built from scratch in Python and PyTorch, running entirely on CPU. No pre-trained models or external APIs used as the model core.

## Architecture

| Component | Specification |
|-----------|--------------|
| Architecture | Decoder-only Transformer |
| Parameters | ~5.3M (configurable, 5-10M range) |
| Layers | 4 Transformer blocks |
| Attention Heads | 4 |
| Hidden Dimension | 256 |
| Feed-Forward Dimension | 1024 |
| Context Length | 512 tokens |
| Tokenizer | BPE (byte-level, trainable, 8K vocab) |
| Embedding | Token + Positional (learnable) |
| Normalization | Pre-LayerNorm |
| Activation | GELU |
| Output | Linear LM head (weight-tied with token embedding) |
| Loss | Cross-Entropy |
| Optimizer | AdamW |
| Training | Real backprop, gradient clipping, train/val loss tracking |
| Checkpoint | Full model + optimizer + step/epoch save/load |
| Inference | Autoregressive with temperature + top-k sampling |
| Hardware | CPU only, no GPU required |

## Project Structure

```
llm/
  config.py       - All hyperparameters in one place (editable)
  tokenizer.py    - BPE tokenizer from scratch (no external libs)
  model.py        - GPT model: attention, feed-forward, transformer blocks
  dataset.py      - Text dataset for next-token prediction
  checkpoint.py   - Save/load full training state
  training.py     - Training loop with AdamW, grad clipping, loss tracking
  inference.py    - Text generation with temperature/top-k sampling
  tests.py        - Correctness tests (8 tests)
  sample_data.py  - Sample Python/JS/TS/React/Next.js code snippets
  main.py         - Entry point: test, train, generate, info
```

## Usage

```bash
# Run correctness tests (forward, loss, gradient, checkpoint, memorization)
python llm/main.py test

# Train on sample programming data
python llm/main.py train

# Generate text from a trained checkpoint
python llm/main.py generate

# Print model architecture and parameter count
python llm/main.py info
```

## Verification

All 8 correctness tests pass:
1. Forward pass produces correct output shape
2. Loss is a finite scalar
3. Backward pass produces gradients on all parameters
4. Weights change after optimizer steps
5. Checkpoint save/load round-trips exactly
6. Tokenizer encode/decode round-trips correctly
7. Model memorizes a tiny dataset (loss drops to <1% of initial)
8. Inference produces valid decodable text

## Configuration

Edit `llm/config.py` to change any hyperparameter: number of layers, model dimensions, vocabulary size, context length, learning rate, batch size, etc. The model is fully modular and adapts automatically.

## Adding Real Training Data

Replace `sample_data.py` with larger code corpora for Python, JavaScript, TypeScript, React, and Next.js. The model and training pipeline are ready — just provide more data and increase `max_epochs`.

## No External Dependencies (for the model)

The model, tokenizer, training loop, and inference are implemented purely in Python + PyTorch. No HuggingFace, no transformers library, no pre-trained weights, no API calls.
