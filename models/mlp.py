import numpy as np


class MLP:

    def __init__(self, input_dim, hidden_dim, output_dim, std=0.01):
        self.W1 = np.random.randn(input_dim, hidden_dim) * std
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.randn(hidden_dim, output_dim) * std
        self.b2 = np.zeros(output_dim)

        # 中间变量
        self.x = None
        self.z1 = None
        self.a1 = None
        self.z2 = None

    # 前向传播（返回 logits）
    def forward(self, X):
        self.x = X

        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    # 交叉熵损失（用 logits 外部算 softmax）
    def loss(self, y_true):
        raise NotImplementedError(
            "Use CrossEntropyLoss instead."
        )

    # 反向传播
    def backward(self, y_true):
        N = y_true.shape[0]

        # softmax + grad
        z2_shifted = self.z2 - np.max(self.z2, axis=1, keepdims=True)
        exp_z2 = np.exp(z2_shifted)
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)

        dZ2 = probs.copy()
        dZ2[np.arange(N), y_true] -= 1
        dZ2 /= N

        dW2 = self.a1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.z1 > 0)

        dW1 = self.x.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        return {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2
        }

    def predict(self, X):
        logits = self.forward(X)
        z2_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_z2 = np.exp(z2_shifted)
        probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)