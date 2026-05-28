import numpy as np


def softmax(x):

    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(probs, y_true):
    N = y_true.shape[0]
    eps = 1e-12
    log_probs = -np.log(
        probs[np.arange(N), y_true.astype(np.int64)] + eps
    )
    return np.mean(log_probs)


class CrossEntropyLoss:

    def __call__(self, logits, y_true):
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, y_true)
        return loss, probs