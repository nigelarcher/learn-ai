# Custom Training Loop Explained

## What This File Does

The `training.py` file implements a custom training loop from scratch, handling:
- Forward propagation through the entire model
- Loss computation (cross-entropy for language modeling)
- Backpropagation through all layers
- Parameter updates with various optimizers
- Learning rate scheduling
- Gradient clipping for stability

## Why It's Important

For health insurance claims processing:

- **Loss Design**: Custom losses can incorporate business rules (e.g., penalize false claim rejections more than false approvals)
- **Training Control**: Fine-grained control over training dynamics for sensitive medical data
- **Debugging**: Understanding exactly how the model learns claim patterns
- **Optimization**: Tune training specifically for medical terminology convergence

## How It Contributes to Learning

Building a custom training loop teaches:

1. **Backpropagation**: Manual implementation of chain rule through complex architectures
2. **Optimizer Mechanics**: How Adam, SGD, and learning rate schedules work
3. **Numerical Stability**: Gradient clipping, loss scaling, and overflow prevention
4. **Training Dynamics**: How loss curves relate to model behavior

## Key Components

### Cross-Entropy Loss
```
L = -1/N Σ log(p_yi)
```
Where p_yi is the predicted probability of the correct token.

### Adam Optimizer
```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
θ_t = θ_{t-1} - α * m_t / (√v_t + ε)
```

### Learning Rate Scheduling
```
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

## Medical Claims Training Considerations

1. **Imbalanced Data**: Most claims are valid; few are fraudulent
2. **Vocabulary**: Medical terms are rare but crucial
3. **Sequence Length**: Claims can be very long documents
4. **Privacy**: Need to train without memorizing specific patient data

## Training Pipeline

1. **Data Loading**: Batch claims with similar lengths
2. **Forward Pass**: Process through transformer
3. **Loss Computation**: Calculate prediction error
4. **Backward Pass**: Compute gradients
5. **Parameter Update**: Apply optimizer
6. **Monitoring**: Track loss, accuracy, and medical term understanding