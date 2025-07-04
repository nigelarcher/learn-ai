import numpy as np


class LayerNormalization:
    """Layer normalization for transformer models."""
    
    def __init__(self, d_model, eps=1e-6):
        """
        Args:
            d_model: Dimension of the model
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        
        self.grad_gamma = None
        self.grad_beta = None
        
    def forward(self, x):
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Normalized output
        """
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        
        self.x_norm = (x - self.mean) / self.std
        
        output = self.gamma * self.x_norm + self.beta
        
        return output
    
    def backward(self, grad_output):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient from next layer
        
        Returns:
            Gradient w.r.t input
        """
        N = self.x.shape[-1]
        
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=(0, 1))
        self.grad_beta = np.sum(grad_output, axis=(0, 1))
        
        grad_x_norm = grad_output * self.gamma
        
        grad_var = np.sum(grad_x_norm * (self.x - self.mean) * -0.5 * (self.var + self.eps)**(-1.5), 
                         axis=-1, keepdims=True)
        
        grad_mean = np.sum(grad_x_norm * -1.0 / self.std, axis=-1, keepdims=True)
        grad_mean += grad_var * np.sum(-2.0 * (self.x - self.mean), axis=-1, keepdims=True) / N
        
        grad_x = grad_x_norm / self.std
        grad_x += grad_var * 2.0 * (self.x - self.mean) / N
        grad_x += grad_mean / N
        
        return grad_x
    
    def update_parameters(self, learning_rate):
        """Update gamma and beta parameters."""
        if self.grad_gamma is not None:
            self.gamma -= learning_rate * self.grad_gamma
            self.beta -= learning_rate * self.grad_beta
            self.grad_gamma = None
            self.grad_beta = None


class RMSNorm:
    """Root Mean Square Layer Normalization (simpler alternative)."""
    
    def __init__(self, d_model, eps=1e-6):
        """
        Args:
            d_model: Dimension of the model
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        
        self.gamma = np.ones(d_model)
        self.grad_gamma = None
        
    def forward(self, x):
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Normalized output
        """
        self.x = x
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.x_norm = x / self.rms
        
        output = self.gamma * self.x_norm
        
        return output
    
    def backward(self, grad_output):
        """
        Compute gradients for backpropagation.
        
        Args:
            grad_output: Gradient from next layer
        
        Returns:
            Gradient w.r.t input
        """
        N = self.x.shape[-1]
        
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=(0, 1))
        
        grad_x_norm = grad_output * self.gamma
        
        grad_rms = -np.sum(grad_x_norm * self.x / (self.rms**2), axis=-1, keepdims=True)
        
        grad_x = grad_x_norm / self.rms
        grad_x += grad_rms * self.x / (N * self.rms)
        
        return grad_x
    
    def update_parameters(self, learning_rate):
        """Update gamma parameter."""
        if self.grad_gamma is not None:
            self.gamma -= learning_rate * self.grad_gamma
            self.grad_gamma = None


class PreLayerNorm:
    """Wrapper for pre-normalization in transformer blocks."""
    
    def __init__(self, d_model, sublayer, eps=1e-6):
        """
        Args:
            d_model: Model dimension
            sublayer: The sublayer (attention or FFN) to wrap
            eps: Epsilon for layer norm
        """
        self.norm = LayerNormalization(d_model, eps)
        self.sublayer = sublayer
        
    def forward(self, x, *args, **kwargs):
        """Apply pre-normalization: norm -> sublayer -> residual."""
        normalized = self.norm.forward(x)
        output = self.sublayer.forward(normalized, *args, **kwargs)
        return x + output
    
    def backward(self, grad_output):
        """Backpropagate through pre-normalization."""
        grad_sublayer = self.sublayer.backward(grad_output)
        grad_norm = self.norm.backward(grad_sublayer)
        return grad_output + grad_norm