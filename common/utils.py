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


def cos_similarity(x, y, eps=1e-8):
    return np.dot(x, y) / (np.linalg.norm(x) + eps) / (np.linalg.norm(y) + eps)


def most_similar(word, words, co_matrix, top=5):
    try:
        word_id = words.index(word)
    except ValueError:
        print(f"'{word}' is not found.")
        return

    print(f'[query]: {word}')

    word_vec = co_matrix[word_id]
    similarity = [[w, cos_similarity(word_vec, co_matrix[i])]
                  for i, w in enumerate(words) if i != word_id]
    similarity.sort(key=lambda sim: sim[1], reverse=True)
    for s in similarity[:top]:
        print(f'  {s[0]}: {s[1]}')
    print()


def analogy(a, b, c, words, co_matrix, top=5, answer=None):
    try:
        a_vec, b_vec, c_vec = \
            co_matrix[[words.index(word) for word in (a, b, c)]]
    except ValueError as err:
        print(err)
        return

    print(f'{a}:{b} = {c}:?')
    query_vec = b_vec - a_vec + c_vec
    if answer is not None:
        try:
            answer_id = words.index(answer)
            print(f'  ==> {answer}: '
                  f'{cos_similarity(co_matrix[answer_id], query_vec)}')
        except ValueError as err:
            print(err)
    similarity = [[w, cos_similarity(query_vec, co_matrix[i])]
                  for i, w in enumerate(words)]
    similarity.sort(key=lambda sim: sim[1], reverse=True)
    for s in similarity[:top]:
        print(f'  {s[0]}: {s[1]}')
    print()
