import numpy as np


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = []
    for word in words:
        if word not in id_to_word:
            word_to_id[word] = len(id_to_word)
            id_to_word.append(word)

    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, window_size=1):
    vocab_size = max(corpus) + 1
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        for i in range(window_size):
            left_idx = idx - i - 1
            right_idx = idx + i + 1
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    return np.dot(x, y) / (np.linalg.norm(x) + eps) / (np.linalg.norm(y) + eps)


def most_similar(word, words, co_matrix, top=5):
    try:
        word_id = words.index(word)
    except ValueError:
        print(f"'{word}' is not found.")
        return

    print(f'\n[query]: {word}')

    word_vec = co_matrix[word_id]
    similarity = [[w, cos_similarity(word_vec, co_matrix[i])]
                  for i, w in enumerate(words) if i != word_id]
    similarity.sort(key=lambda sim: sim[1], reverse=True)
    for s in similarity[:top]:
        print(f' {s[0]}: {s[1]}')


def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    n = np.sum(C)
    s = np.sum(C, axis=0)
    cnt = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] != 0:
                print(s[i] * s[j] + eps)
                pmi = np.log2(C[i, j] * n / (s[i] * s[j] + eps))
                M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % max(1, (C.size // 100)) == 0:
                    print(f'{(100 * cnt) // C.size}% done.')
    return M
