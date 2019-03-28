import collections
from functools import reduce

import numpy as np

from common.layers import SigmoidWithLoss, EmbeddingDot


class UnigramSampler:
    def __init__(self, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        counts = collections.Counter(corpus)
        self.vocab_size = max(counts) + 1
        self.word_p = np.zeros(self.vocab_size)
        for word_id, count in counts.items():
            self.word_p[word_id] = count
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        negative_sample = np.zeros((batch_size, self.sample_size),
                                   dtype=np.int32)

        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size,
                                                     size=self.sample_size,
                                                     replace=False,
                                                     p=p)
        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W)
                                 for _ in range(sample_size + 1)]

        self.params, self.grads = reduce(
            lambda pg, layer: [pg[0] + layer.params, pg[1] + layer.grads],
            self.embed_dot_layers,
            [[], []]
        )

    def forward(self, h, target):
        batch_size = target.shape[0]

        loss = self.loss_layers[0].forward(
            self.embed_dot_layers[0].forward(h, target),
            np.ones(batch_size, dtype=np.int32),
        )

        negative_sample = self.sampler.get_negative_sample(target)
        negative_label = np.zeros(batch_size, dtype=np.int32)
        return reduce(
            lambda l, i: l + self.loss_layers[i + 1].forward(
                self.embed_dot_layers[i + 1].forward(h, negative_sample[:, i]),
                negative_label,
            ),
            range(self.sample_size),
            loss,
        )

    def backward(self, dout=1):
        return reduce(
            lambda dh, ls: dh + ls[1].backward(ls[0].backward(dout)),
            zip(self.loss_layers, self.embed_dot_layers),
            0,
        )
