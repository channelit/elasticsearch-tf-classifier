from elasticsearch import Elasticsearch
import nltk


docs = docs()
tokens = tokens(docs)

import gensim
model = gensim.models.Word2Vec(tokens, size=100)

model.most_similar(positive=['method'])

w2v = model.wv
# w2v = dict(zip(model.wv.index2word, model.wv.syn0))
