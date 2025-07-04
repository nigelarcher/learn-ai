import numpy as np


class PositionalEncoding:
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model, max_seq_length=5000):
        """
        Args:
            d_model: Dimension of the model
            max_seq_length: Maximum sequence length to pre-compute
        """
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        self.encoding = self._create_encoding()
        
    def _create_encoding(self):
        """Create sinusoidal position encoding."""
        encoding = np.zeros((self.max_seq_length, self.d_model))
        position = np.arange(self.max_seq_length).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        encoding[:, 0::2] = np.sin(position * div_term)
        encoding[:, 1::2] = np.cos(position * div_term)
        
        return encoding
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            x + positional encoding
        """
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len, :]
    
    def get_encoding(self, positions):
        """
        Get positional encoding for specific positions.
        
        Args:
            positions: Array of position indices
        
        Returns:
            Positional encodings for the given positions
        """
        return self.encoding[positions]


class LearnedPositionalEncoding:
    """Learned positional embeddings (alternative to sinusoidal)."""
    
    def __init__(self, d_model, max_seq_length=5000):
        """
        Args:
            d_model: Dimension of the model
            max_seq_length: Maximum sequence length
        """
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        self.embeddings = np.random.randn(max_seq_length, d_model) * 0.02
        self.grad_embeddings = None
        
    def forward(self, x):
        """
        Add learned positional embeddings to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            x + positional embeddings
        """
        seq_len = x.shape[1]
        self.positions_used = np.arange(seq_len)
        return x + self.embeddings[:seq_len, :]
    
    def backward(self, grad_output):
        """
        Compute gradients for learned embeddings.
        
        Args:
            grad_output: Gradient from next layer [batch_size, seq_len, d_model]
        """
        if self.grad_embeddings is None:
            self.grad_embeddings = np.zeros_like(self.embeddings)
        
        batch_size = grad_output.shape[0]
        for pos in self.positions_used:
            self.grad_embeddings[pos] += np.sum(grad_output[:, pos, :], axis=0)
        
        return grad_output
    
    def update(self, learning_rate):
        """Update learned embeddings using accumulated gradients."""
        if self.grad_embeddings is not None:
            self.embeddings -= learning_rate * self.grad_embeddings
            self.grad_embeddings = None


class RelativePositionalEncoding:
    """Relative positional encoding for capturing relative distances."""
    
    def __init__(self, d_model, max_relative_position=128):
        """
        Args:
            d_model: Dimension of the model
            max_relative_position: Maximum relative distance to consider
        """
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        self.relative_embeddings = np.random.randn(
            2 * max_relative_position + 1, d_model
        ) * 0.02
        
    def get_relative_positions(self, seq_len):
        """
        Create relative position matrix.
        
        Args:
            seq_len: Sequence length
        
        Returns:
            Matrix of relative positions [seq_len, seq_len]
        """
        positions = np.arange(seq_len)
        relative_positions = positions[:, None] - positions[None, :]
        
        relative_positions = np.clip(
            relative_positions,
            -self.max_relative_position,
            self.max_relative_position
        )
        
        relative_positions += self.max_relative_position
        
        return relative_positions
    
    def get_embeddings(self, seq_len):
        """Get relative positional embeddings for a sequence."""
        relative_positions = self.get_relative_positions(seq_len)
        return self.relative_embeddings[relative_positions]