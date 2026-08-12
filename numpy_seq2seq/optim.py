"""Adam optimizer and gradient clipping, implemented directly over a plain
dict of NumPy arrays (no framework, no autograd)."""
import numpy as np


def clip_grads_(grads, max_norm=5.0):
    """In-place global-norm clipping across every array in the grads dict."""
    total_sq = 0.0
    for g in grads.values():
        total_sq += float(np.sum(g * g))
    total_norm = np.sqrt(total_sq)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for k in grads:
            grads[k] *= scale
    return total_norm


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def set_lr(self, lr):
        """Used by the training loop to decay the learning rate on plateau."""
        self.lr = lr

    def step(self, params, grads):
        self.t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps
        for k in params:
            g = grads[k]
            # Weight decay pulls weights toward zero each step, discouraging
            # the big weights a memorizing model develops. Biases are left
            # alone - shrinking them just shifts the function, it does not
            # reduce capacity.
            if self.weight_decay > 0.0 and params[k].ndim > 1:
                g = g + self.weight_decay * params[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            m_hat = self.m[k] / (1 - b1 ** self.t)
            v_hat = self.v[k] / (1 - b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
