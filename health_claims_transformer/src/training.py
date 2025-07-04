import numpy as np
from transformer import Transformer


def cross_entropy_loss(logits, targets, ignore_index=-100):
    """
    Compute cross-entropy loss for language modeling.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        targets: True labels [batch_size, seq_len]
        ignore_index: Index to ignore in loss computation
    
    Returns:
        loss: Scalar loss value
        grad_logits: Gradient w.r.t logits
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    mask = targets_flat != ignore_index
    valid_targets = targets_flat[mask]
    valid_logits = logits_flat[mask]
    
    if len(valid_targets) == 0:
        return 0.0, np.zeros_like(logits)
    
    exp_logits = np.exp(valid_logits - np.max(valid_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    correct_probs = probs[np.arange(len(valid_targets)), valid_targets]
    loss = -np.mean(np.log(correct_probs + 1e-8))
    
    grad_probs = probs.copy()
    grad_probs[np.arange(len(valid_targets)), valid_targets] -= 1
    grad_probs /= len(valid_targets)
    
    grad_logits = np.zeros((batch_size * seq_len, vocab_size))
    grad_logits[mask] = grad_probs
    grad_logits = grad_logits.reshape(batch_size, seq_len, vocab_size)
    
    return loss, grad_logits


class AdamOptimizer:
    """Adam optimizer implementation."""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}
        
    def update(self, params, grads):
        """
        Update parameters using Adam optimization.
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        self.t += 1
        
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class WarmupScheduler:
    """Learning rate scheduler with linear warmup."""
    
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step = 0
        
    def get_lr(self):
        """Get current learning rate."""
        self.step += 1
        arg1 = self.step ** (-0.5)
        arg2 = self.step * (self.warmup_steps ** (-1.5))
        return self.d_model ** (-0.5) * min(arg1, arg2)


class Trainer:
    """Custom training loop for transformer model."""
    
    def __init__(self, model, optimizer, scheduler=None, grad_clip=1.0):
        """
        Args:
            model: Transformer model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            grad_clip: Maximum gradient norm
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.training_step = 0
        
    def clip_gradients(self, gradients, max_norm):
        """Clip gradients by global norm."""
        total_norm = 0
        for grad in gradients.values():
            if grad is not None:
                total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for key in gradients:
                if gradients[key] is not None:
                    gradients[key] *= clip_coef
        
        return total_norm
    
    def train_step(self, src, tgt, tgt_labels):
        """
        Single training step.
        
        Args:
            src: Source sequences [batch_size, src_len]
            tgt: Target sequences (input) [batch_size, tgt_len]
            tgt_labels: Target sequences (labels) [batch_size, tgt_len]
        
        Returns:
            loss: Training loss
            accuracy: Token accuracy
        """
        logits = self.model.forward(src, tgt, training=True)
        
        loss, grad_logits = cross_entropy_loss(logits, tgt_labels)
        
        gradients = self._backward(grad_logits)
        
        grad_norm = self.clip_gradients(gradients, self.grad_clip)
        
        if self.scheduler:
            self.optimizer.learning_rate = self.scheduler.get_lr()
        
        params = self._get_all_params()
        self.optimizer.update(params, gradients)
        self._set_all_params(params)
        
        predictions = np.argmax(logits, axis=-1)
        mask = tgt_labels != -100
        accuracy = np.sum((predictions == tgt_labels) * mask) / np.sum(mask)
        
        self.training_step += 1
        
        return loss, accuracy, grad_norm
    
    def _backward(self, grad_output):
        """Backpropagate through entire model."""
        gradients = {}
        
        grad = grad_output
        
        for i in reversed(range(self.model.n_layers)):
            grad = self.model.decoder_blocks[i].backward(grad)
        
        gradients['token_embedding'] = np.zeros_like(self.model.token_embedding)
        
        return gradients
    
    def _get_all_params(self):
        """Get all model parameters."""
        params = {
            'token_embedding': self.model.token_embedding,
            'output_projection': self.model.output_projection,
        }
        
        for i, block in enumerate(self.model.encoder_blocks):
            params[f'encoder_{i}_attention_W_q'] = block.attention.W_q
            params[f'encoder_{i}_attention_W_k'] = block.attention.W_k
            params[f'encoder_{i}_attention_W_v'] = block.attention.W_v
            params[f'encoder_{i}_attention_W_o'] = block.attention.W_o
            
            params[f'encoder_{i}_ffn_W1'] = block.feed_forward.W1
            params[f'encoder_{i}_ffn_b1'] = block.feed_forward.b1
            params[f'encoder_{i}_ffn_W2'] = block.feed_forward.W2
            params[f'encoder_{i}_ffn_b2'] = block.feed_forward.b2
            
            params[f'encoder_{i}_norm1_gamma'] = block.norm1.gamma
            params[f'encoder_{i}_norm1_beta'] = block.norm1.beta
            params[f'encoder_{i}_norm2_gamma'] = block.norm2.gamma
            params[f'encoder_{i}_norm2_beta'] = block.norm2.beta
        
        return params
    
    def _set_all_params(self, params):
        """Set all model parameters from dictionary."""
        self.model.token_embedding = params['token_embedding']
        self.model.output_projection = params['output_projection']
        
        for i, block in enumerate(self.model.encoder_blocks):
            block.attention.W_q = params[f'encoder_{i}_attention_W_q']
            block.attention.W_k = params[f'encoder_{i}_attention_W_k']
            block.attention.W_v = params[f'encoder_{i}_attention_W_v']
            block.attention.W_o = params[f'encoder_{i}_attention_W_o']
            
            block.feed_forward.W1 = params[f'encoder_{i}_ffn_W1']
            block.feed_forward.b1 = params[f'encoder_{i}_ffn_b1']
            block.feed_forward.W2 = params[f'encoder_{i}_ffn_W2']
            block.feed_forward.b2 = params[f'encoder_{i}_ffn_b2']
            
            block.norm1.gamma = params[f'encoder_{i}_norm1_gamma']
            block.norm1.beta = params[f'encoder_{i}_norm1_beta']
            block.norm2.gamma = params[f'encoder_{i}_norm2_gamma']
            block.norm2.beta = params[f'encoder_{i}_norm2_beta']
    
    def validate(self, val_loader):
        """Run validation on a dataset."""
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        for src, tgt, tgt_labels in val_loader:
            logits = self.model.forward(src, tgt, training=False)
            loss, _ = cross_entropy_loss(logits, tgt_labels)
            
            predictions = np.argmax(logits, axis=-1)
            mask = tgt_labels != -100
            accuracy = np.sum((predictions == tgt_labels) * mask) / np.sum(mask)
            
            batch_size = src.shape[0]
            total_loss += loss * batch_size
            total_accuracy += accuracy * batch_size
            total_samples += batch_size
        
        return total_loss / total_samples, total_accuracy / total_samples


def create_sample_data(batch_size=32, seq_len=128, vocab_size=1000):
    """Create sample data for testing."""
    src = np.random.randint(1, vocab_size, (batch_size, seq_len))
    tgt = np.random.randint(1, vocab_size, (batch_size, seq_len))
    tgt_labels = np.concatenate([tgt[:, 1:], np.ones((batch_size, 1), dtype=int) * -100], axis=1)
    
    return src, tgt, tgt_labels