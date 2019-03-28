import numpy as np

from .functions import softmax, cross_entropy_error, sigmoid


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class BaseWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def _output(self, x):
        raise NotImplementedError()

    def _pre_forward(self, x, t):
        self.t = t
        self.y = self._output(x)

    def _output_error(self, x, t):
        raise NotImplementedError()

    def forward(self, x, t):
        self._pre_forward(x, t)
        return self._output_error(x, t)


class SoftmaxWithLoss(BaseWithLoss):
    def _output(self, x):
        return softmax(x)

    def _output_error(self, x, t):
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
        return cross_entropy_error(self.y, self.t)

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        return dx / batch_size


class SigmoidWithLoss(BaseWithLoss):
    def _output(self, x):
        return sigmoid(x)

    def _output_error(self, x, t):
        return cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t) * dout / batch_size


class Embedding(MatMul):
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        return W[idx]

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        self.cache = (h, target_W)
        return np.sum(target_W * h, axis=1)

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        return dout * target_W
