# Attention Mechanism Explained

## What This File Does

The `attention.py` file implements the core attention mechanism of the transformer architecture from scratch using only NumPy. It includes:

1. **Scaled Dot-Product Attention**: The fundamental operation that allows the model to focus on different parts of the input
2. **Multi-Head Attention**: Multiple attention mechanisms running in parallel to capture different types of relationships

## Why It's Important

Attention is the breakthrough that made transformers revolutionary. For health insurance claims:

- **Context Understanding**: Attention allows the model to understand relationships between medical terms and policy conditions across long documents
- **Pattern Recognition**: It can identify which parts of a claim are most relevant to specific policy clauses
- **Anomaly Detection**: By learning normal attention patterns, it can spot unusual claim structures that might indicate fraud

## How It Contributes to Learning

Building attention from scratch teaches:

1. **Matrix Operations**: Understanding Q (Query), K (Key), and V (Value) matrices
2. **Computational Complexity**: O(n²) scaling with sequence length
3. **Parallelization**: Why attention is more efficient than RNNs
4. **Hyperparameter Impact**: How embedding dimensions and head count affect performance

## Key Formulas

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q: Query matrix (what we're looking for)
- K: Key matrix (what we compare against)
- V: Value matrix (what we actually use)
- d_k: Dimension of key vectors (for scaling)

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

## Implementation Challenges

1. **Numerical Stability**: Preventing overflow in softmax with large dot products
2. **Memory Efficiency**: Managing O(n²) memory requirements for attention scores
3. **Gradient Flow**: Ensuring stable backpropagation through attention layers