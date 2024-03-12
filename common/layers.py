import numpy as np
from functions import sigmoid
from functions import cross_entropy_error


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        out = np.dot(x, self.W) +self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None    # 損失
        self.y = None       # sigmoid関数の出力
        self.t = None       # 教師データ (one-hot)

    def forward(self, x):
        self.y = sigmoid(x)
        loss = cross_entropy_error(self.y, self.t)

        return loss

    def backward(self, t):
        batch_size = np.shape[0]
        dx = (self.y - self.t) / batch_size


