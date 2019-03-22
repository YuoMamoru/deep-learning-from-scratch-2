from common.trainer import Trainer
from common.optimizer import Adam
from .simple_cbow import SimpleCBOW
from common.utils import preprocess, create_contexts_target, convert_one_hot


window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodby and I say hello.'
corpus, words = preprocess(text)

contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, len(words))
contexts = convert_one_hot(contexts, len(words))

model = SimpleCBOW(len(words), hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

for i, word in enumerate(words):
    print(word, model.word_vecs[i])
