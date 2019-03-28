import pickle

import numpy as np

from common.trainer import Trainer
from common.optimizer import Adam
from .cbow import CBOW
from common.utils import create_contexts_target  # , convert_one_hot
from dataset import ptb


window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

corpus, word_to_id, words = ptb.load_data('train')
vocab_size = len(words)

contexts, target = create_contexts_target(corpus, window_size)
# target = convert_one_hot(target, len(words))
# contexts = convert_one_hot(contexts, len(words))

model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['words'] = words
pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
