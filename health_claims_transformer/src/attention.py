import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class ScaledDotProductAttention:
    """Scaled dot-product attention mechanism."""
    
    def __init__(self, d_k):
        self.d_k = d_k
        self.scale = np.sqrt(d_k)
        
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query matrix [batch_size, seq_len, d_k]
            K: Key matrix [batch_size, seq_len, d_k]
            V: Value matrix [batch_size, seq_len, d_v]
            mask: Optional mask [batch_size, seq_len, seq_len]
        
        Returns:
            output: Attention output [batch_size, seq_len, d_v]
            attention_weights: Attention weights [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = Q.shape
        
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / self.scale
        
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        
        attention_weights = softmax(scores, axis=-1)
        
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def backward(self, grad_output, Q, K, V, attention_weights):
        """Compute gradients for backpropagation."""
        grad_V = np.matmul(attention_weights.transpose(0, 2, 1), grad_output)
        
        grad_attention = np.matmul(grad_output, V.transpose(0, 2, 1))
        
        grad_scores = attention_weights * (grad_attention - 
                      np.sum(grad_attention * attention_weights, axis=-1, keepdims=True))
        
        grad_scores = grad_scores / self.scale
        
        grad_Q = np.matmul(grad_scores, K)
        grad_K = np.matmul(grad_scores.transpose(0, 2, 1), Q)
        
        return grad_Q, grad_K, grad_V


class MultiHeadAttention:
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
        
        self.grad_W_q = None
        self.grad_W_k = None
        self.grad_W_v = None
        self.grad_W_o = None
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: Optional mask
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Debug shape issues
        if not isinstance(query, np.ndarray):
            raise TypeError(f"Expected numpy array for query, got {type(query)}")
        if len(query.shape) != 3:
            raise ValueError(f"Expected 3D query tensor, got shape {query.shape}")
            
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_kv, _ = key.shape
        
        Q = np.matmul(query, self.W_q).reshape(batch_size, seq_len_q, self.n_heads, self.d_k)
        K = np.matmul(key, self.W_k).reshape(batch_size, seq_len_kv, self.n_heads, self.d_k)
        V = np.matmul(value, self.W_v).reshape(batch_size, seq_len_kv, self.n_heads, self.d_v)
        
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        attention_outputs = []
        attention_weights_list = []
        
        for i in range(self.n_heads):
            output, weights = self.attention.forward(Q[:, i], K[:, i], V[:, i], mask)
            attention_outputs.append(output)
            attention_weights_list.append(weights)
        
        concat_attention = np.concatenate(attention_outputs, axis=-1)
        
        output = np.matmul(concat_attention, self.W_o)
        
        self.cache = (query, key, value, Q, K, V, attention_outputs, attention_weights_list)
        
        return output
    
    def backward(self, grad_output):
        """Compute gradients for all parameters."""
        query, key, value, Q, K, V, attention_outputs, attention_weights_list = self.cache
        batch_size, seq_len, _ = query.shape
        
        self.grad_W_o = np.matmul(
            np.concatenate(attention_outputs, axis=-1).reshape(-1, self.d_model).T,
            grad_output.reshape(-1, self.d_model)
        )
        
        grad_concat = np.matmul(grad_output, self.W_o.T)
        
        grad_heads = np.split(grad_concat, self.n_heads, axis=-1)
        
        grad_Q_all = []
        grad_K_all = []
        grad_V_all = []
        
        for i in range(self.n_heads):
            grad_Q, grad_K, grad_V = self.attention.backward(
                grad_heads[i], Q[:, i], K[:, i], V[:, i], attention_weights_list[i]
            )
            grad_Q_all.append(grad_Q)
            grad_K_all.append(grad_K)
            grad_V_all.append(grad_V)
        
        grad_Q = np.stack(grad_Q_all, axis=1).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        grad_K = np.stack(grad_K_all, axis=1).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        grad_V = np.stack(grad_V_all, axis=1).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        self.grad_W_q = np.matmul(query.reshape(-1, self.d_model).T, grad_Q.reshape(-1, self.d_model))
        self.grad_W_k = np.matmul(key.reshape(-1, self.d_model).T, grad_K.reshape(-1, self.d_model))
        self.grad_W_v = np.matmul(value.reshape(-1, self.d_model).T, grad_V.reshape(-1, self.d_model))
        
        grad_query = np.matmul(grad_Q, self.W_q.T)
        grad_key = np.matmul(grad_K, self.W_k.T)
        grad_value = np.matmul(grad_V, self.W_v.T)
        
        return grad_query, grad_key, grad_value