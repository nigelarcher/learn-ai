import numpy as np
from attention import MultiHeadAttention
from positional_encoding import PositionalEncoding
from layer_norm import LayerNormalization


def gelu(x):
    """Gaussian Error Linear Unit activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def gelu_derivative(x):
    """Derivative of GELU for backpropagation."""
    tanh_arg = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    tanh_val = np.tanh(tanh_arg)
    sech_squared = 1 - tanh_val**2
    return 0.5 * (1 + tanh_val) + 0.5 * x * sech_squared * np.sqrt(2 / np.pi) * (1 + 0.134145 * x**2)


class FeedForward:
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (typically 4 * d_model)
            dropout_rate: Dropout probability
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
        
        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None
        
    def forward(self, x, training=True):
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            training: Whether in training mode (for dropout)
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        self.x = x
        self.hidden = np.matmul(x, self.W1) + self.b1
        self.activated = gelu(self.hidden)
        
        if training and self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, self.activated.shape)
            self.activated *= self.dropout_mask / (1 - self.dropout_rate)
        
        output = np.matmul(self.activated, self.W2) + self.b2
        
        return output
    
    def backward(self, grad_output):
        """Compute gradients for backpropagation."""
        batch_size, seq_len, _ = grad_output.shape
        
        self.grad_W2 = np.matmul(
            self.activated.reshape(-1, self.d_ff).T,
            grad_output.reshape(-1, self.d_model)
        )
        self.grad_b2 = np.sum(grad_output, axis=(0, 1))
        
        grad_hidden = np.matmul(grad_output, self.W2.T)
        
        if hasattr(self, 'dropout_mask'):
            grad_hidden *= self.dropout_mask / (1 - self.dropout_rate)
        
        grad_activated = grad_hidden * gelu_derivative(self.hidden)
        
        self.grad_W1 = np.matmul(
            self.x.reshape(-1, self.d_model).T,
            grad_activated.reshape(-1, self.d_ff)
        )
        self.grad_b1 = np.sum(grad_activated, axis=(0, 1))
        
        grad_x = np.matmul(grad_activated, self.W1.T)
        
        return grad_x


class EncoderBlock:
    """Single encoder block of the transformer."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout_rate: Dropout probability
        """
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout_rate = dropout_rate
        
    def forward(self, x, mask=None, training=True):
        """
        Forward pass through encoder block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask
            training: Whether in training mode
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        normed = self.norm1.forward(x)
        attention_output = self.attention.forward(normed, normed, normed, mask)
        
        if training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, attention_output.shape)
            attention_output *= dropout_mask / (1 - self.dropout_rate)
        
        x = x + attention_output
        
        normed = self.norm2.forward(x)
        ff_output = self.feed_forward.forward(normed, training)
        
        if training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, ff_output.shape)
            ff_output *= dropout_mask / (1 - self.dropout_rate)
        
        output = x + ff_output
        
        self.cache = (x, attention_output, ff_output)
        
        return output
    
    def backward(self, grad_output):
        """Backpropagate through encoder block."""
        x, attention_output, ff_output = self.cache
        
        grad_ff = grad_output
        grad_ff_norm = self.feed_forward.backward(grad_ff)
        grad_x2 = self.norm2.backward(grad_ff_norm)
        grad_residual2 = grad_output + grad_x2
        
        grad_attention = grad_residual2
        grad_q, grad_k, grad_v = self.attention.backward(grad_attention)
        grad_attention_norm = grad_q + grad_k + grad_v
        grad_x1 = self.norm1.backward(grad_attention_norm)
        grad_residual1 = grad_residual2 + grad_x1
        
        return grad_residual1


class DecoderBlock:
    """Single decoder block of the transformer."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout_rate: Dropout probability
        """
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout_rate = dropout_rate
        
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None, training=True):
        """
        Forward pass through decoder block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            encoder_output: Output from encoder [batch_size, seq_len, d_model]
            self_mask: Self-attention mask (causal)
            cross_mask: Cross-attention mask
            training: Whether in training mode
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        normed = self.norm1.forward(x)
        self_attention_output = self.self_attention.forward(normed, normed, normed, self_mask)
        
        if training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, self_attention_output.shape)
            self_attention_output *= dropout_mask / (1 - self.dropout_rate)
        
        x = x + self_attention_output
        
        normed = self.norm2.forward(x)
        # Debug: Check shapes
        if isinstance(encoder_output, tuple):
            encoder_output = encoder_output[0]
        cross_attention_output = self.cross_attention.forward(
            normed, encoder_output, encoder_output, cross_mask
        )
        
        if training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, cross_attention_output.shape)
            cross_attention_output *= dropout_mask / (1 - self.dropout_rate)
        
        x = x + cross_attention_output
        
        normed = self.norm3.forward(x)
        ff_output = self.feed_forward.forward(normed, training)
        
        if training and self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, ff_output.shape)
            ff_output *= dropout_mask / (1 - self.dropout_rate)
        
        output = x + ff_output
        
        return output


class Transformer:
    """Complete transformer model."""
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, 
                 max_seq_length=512, dropout_rate=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder/decoder layers
            d_ff: Feed-forward dimension
            max_seq_length: Maximum sequence length
            dropout_rate: Dropout probability
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.token_embedding = np.random.randn(vocab_size, d_model) * np.sqrt(2.0 / vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_blocks = [
            EncoderBlock(d_model, n_heads, d_ff, dropout_rate) 
            for _ in range(n_layers)
        ]
        
        self.decoder_blocks = [
            DecoderBlock(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ]
        
        self.final_norm = LayerNormalization(d_model)
        self.output_projection = np.random.randn(d_model, vocab_size) * np.sqrt(2.0 / d_model)
        
    def create_padding_mask(self, x, pad_idx=0):
        """Create padding mask for attention."""
        return (x == pad_idx)[:, None, None, :]
    
    def create_causal_mask(self, seq_len):
        """Create causal mask for decoder self-attention."""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask[None, None, :, :]
    
    def forward(self, src, tgt, training=True):
        """
        Forward pass through transformer.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            training: Whether in training mode
        
        Returns:
            Output logits [batch_size, tgt_len, vocab_size]
        """
        src_embeddings = self.token_embedding[src]
        src_embeddings = self.positional_encoding.forward(src_embeddings)
        
        encoder_output = src_embeddings
        for encoder_block in self.encoder_blocks:
            encoder_output = encoder_block.forward(encoder_output, training=training)
        
        tgt_embeddings = self.token_embedding[tgt]
        tgt_embeddings = self.positional_encoding.forward(tgt_embeddings)
        
        tgt_mask = self.create_causal_mask(tgt.shape[1])
        
        decoder_output = tgt_embeddings
        for decoder_block in self.decoder_blocks:
            decoder_output = decoder_block.forward(
                decoder_output, encoder_output, self_mask=tgt_mask, training=training
            )
        
        output = self.final_norm.forward(decoder_output)
        logits = np.matmul(output, self.output_projection)
        
        return logits