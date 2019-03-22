from functools import reduce

import numpy as np


def preprocess(text):
    """Preprocess linguistic analysis

    Args:
        text: Text to be analyzed.

    Returns:
        numpy.array: Corpus.
        list: List of words in the text.
    """
    text = text.lower()
    text = text.replace('.', ' .')
    duplicate_words = text.split(' ')
    words = list(set(duplicate_words))
    corpus = np.array([words.index(w) for w in duplicate_words])
    return corpus, words


def create_contexts_target(corpus, window_size=1):
    '''
    '''
    def append_contexts(contexts, i):
        contexts.append(
            corpus[i:i+window_size] + corpus[i+window_size+1:i+2*window_size+1]
        )
        return contexts

    corpus = list(corpus)
    target = corpus[window_size:-window_size]
    contexts = reduce(append_contexts, range(len(target)), [])
    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    shape = list(corpus.shape)
    shape.append(vocab_size)
    one_hot = np.zeros(shape, dtype=np.int32)

    def convert(one_hot, corpus):
        if corpus.ndim == 1:
            for i, value in enumerate(corpus):
                one_hot[i][value] = 1
        else:
            for i, value in enumerate(corpus):
                convert(one_hot[i], value)

    convert(one_hot, corpus)
    return one_hot


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
