from functools import reduce

import numpy as np

from common.layers import Embedding
from .layers import NegativeSamplingLoss


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        W_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype('f')
        W_out = 0.01 * np.random.randn(vocab_size, hidden_size).astype('f')

        self.in_layers = reduce(lambda layers, i: layers + [Embedding(W_in)],
                                range(window_size * 2),
                                [])
        self.ns_loss = NegativeSamplingLoss(W_out, corpus)

        self.params, self.grads = reduce(
            lambda pg, layer: [pg[0] + layer.params, pg[1] + layer.grads],
            self.in_layers + [self.ns_loss],
            [[], []],
        )
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = reduce(lambda h, i_l: h + i_l[1].forward(contexts[:, i_l[0]]),
                   enumerate(self.in_layers),
                   0) / len(self.in_layers)
        return self.ns_loss.forward(h, target)

    def backward(self, dout=1):
        d = self.ns_loss.backward(dout) / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(d)
