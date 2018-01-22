from elasticsearch import Elasticsearch
import nltk

nltk.download('punkt')

from nltk.tokenize import word_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

es = Elasticsearch(["elasticsearch"], maxsize=25)

def tokenize(text):
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
        return 'NC'

def docs():
    docs = []
    res = es.search(index="intelligence", size=15, body={"query": {"match_all": {}}})
    print("Got %d Hits:" % res['hits']['total'])
    for hit in res['hits']['hits']:
        # print("%(category)s %(text)s" % hit["_source"])
        text = hit["_source"]["text"][0]
        text = text.replace('\\n', ' ')
        docs.append(text)
    return docs

def tokens(docs):
    tokens = []
    for doc in docs:
        tokens.append(tokenize(doc))
    return tokens

docs = docs()
tokens = tokens(docs)

import gensim
model = gensim.models.Word2Vec(tokens, size=100)

model.most_similar(positive=['method'])

w2v = model.wv
# w2v = dict(zip(model.wv.index2word, model.wv.syn0))
