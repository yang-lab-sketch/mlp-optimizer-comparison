import numpy as np

class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = {}   # 速度

    def step(self, params, grads):
        # 初始化速度
        if not self.v:
            for key in params:
                self.v[key] = np.zeros_like(params[key])

        for key in params:
            self.v[key] = (
                self.beta * self.v[key]
                + (1 - self.beta) * grads[key]
            )
            params[key] -= self.lr * self.v[key]