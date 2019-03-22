from functools import reduce

import numpy as np

from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        # Initial weight
        W_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype('f')
        W_out = 0.01 * np.random.randn(hidden_size, vocab_size).astype('f')

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()
        self.params, self.grads = reduce(
            lambda args, layer: [args[0] + layer.params, args[1] + layer.grads],
            [self.in_layer0, self.in_layer1, self.out_layer],
            [[], []],
        )
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) / 2
        score = self.out_layer.forward(h)
        return self.loss_layer.forward(score, target)

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds) / 2
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
