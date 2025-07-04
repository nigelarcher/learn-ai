# Health Insurance Claims Transformer

A transformer model built from scratch (no libraries) for processing health insurance claims, implementing attention mechanisms, positional encoding, and custom training loops.

## Project Structure

- `src/` - Core transformer implementation
  - `attention.py` - Multi-head attention mechanism
  - `positional_encoding.py` - Positional encoding implementation
  - `layer_norm.py` - Layer normalization
  - `transformer.py` - Complete transformer model
  - `training.py` - Custom training loop
  - `tokenizer.py` - Medical terminology tokenizer
  - `claims_processor.py` - Claims assessment and anomaly detection

- `data/` - Training and test data
- `models/` - Saved model weights
- `docs/` - Documentation on architecture and scaling

## Key Features

- Pure NumPy implementation (no PyTorch/TensorFlow)
- Specialized for medical terminology and insurance policy language
- Automated claim assessment
- Anomaly detection for fraud/error identification
- Custom tokenizer for medical terms

## Learning Objectives

1. Understand transformer architecture fundamentals
2. Implement attention mechanisms from scratch
3. Build custom training loops
4. Apply transformers to domain-specific language (medical/insurance)
5. Understand computational requirements and scaling challenges