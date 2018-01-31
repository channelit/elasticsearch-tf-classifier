from elasticsearch import Elasticsearch
import nltk
import gensim
import os
import re

nltk.download('punkt')

START_DOC = 5
TRAIN_DOCS = 50

from nltk.tokenize import word_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
from gensim.models.doc2vec import TaggedDocument

es = Elasticsearch(["elasticsearch"], maxsize=25)

taggeddoc = []

def es_docs():
    res = es.search(index="intelligence", size=TRAIN_DOCS, body={"query": {"match_all": {}}})
    print("Got %d Hits:" % res['hits']['total'])
    for hit in res['hits']['hits']:
        # print("%(category)s %(text)s" % hit["_source"])
        text = hit["_source"]["text"][0]
        id = hit["_id"]
        yield text, id

def clean_tokens(text):
    try:
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        import string
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        return words
    except:
        return 'NC', -1

def train():
    for doc, id in es_docs():
        tokens = clean_tokens(doc)
        td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(tokens))).split(),str(id))
        taggeddoc.append(td)

    print ('Data Loading finished')
    print (len(taggeddoc),type(taggeddoc))

    model = gensim.models.Doc2Vec(taggeddoc, dm = 0, alpha=0.025, size= 20, min_alpha=0.025, min_count=0)
    # start training
    for epoch in range(200):
        if epoch % 20 == 0:
            print ('Now training epoch %s'%epoch)
        model.train(taggeddoc,total_examples=model.corpus_count,epochs=model.iter)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    # shows the similar words
    print (model.most_similar('suppli'))

    # shows the learnt embedding
    print (model['suppli'])

if __name__ == '__main__':
    train()