from elasticsearch import Elasticsearch
import nltk
import gensim
import os
import re
from gensim.models.doc2vec import TaggedDocument
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

START_DOC = 5
TRAIN_DOCS = 50

class ElasticSimilarity:

    def __init__(self):

        self.base_dir = '/assets'
        self.model_file = os.path.join(self.base_dir, 'doc_model')
        #open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), 'rt', encoding='utf8')

        self.es = Elasticsearch(["elasticsearch"], maxsize=25)

        self.taggeddoc = []

    def es_docs(self, start_doc, train_docs):
        res = self.es.search(index="intelligence", from_=start_doc, size=train_docs, body={"query": {"match_all": {}}})
        print("Got %d Hits:" % res['hits']['total'])
        for hit in res['hits']['hits']:
            # print("%(category)s %(text)s" % hit["_source"])
            text = hit["_source"]["text"][0]
            text = text.replace('\\n', ' ')
            id = hit["_id"]
            yield text, id

    def clean_tokens(self, text):
        try:
            tokens = word_tokenize(text)
            tokens = [w.lower() for w in tokens]
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            words = [word for word in stripped if word.isalpha()]
            words = [w for w in words if (len(w) > 2 and not w in stop_words)]
            words = [porter.stem(w) for w in words]
            return words
        except:
            return 'NC'

    def train(self, start_doc, train_docs):
        for doc, id in self.es_docs(start_doc, train_docs):
            tokens = self.clean_tokens(doc)
            if tokens != 'NC':
                td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(tokens))).split(),str(id))
                self.taggeddoc.append(td)

        print ('Data Loading finished')
        print (len(self.taggeddoc),type(self.taggeddoc))

        model = gensim.models.Doc2Vec(self.taggeddoc, dm = 0, alpha=0.025, size= 20, min_alpha=0.025, min_count=0)
        # start training
        for epoch in range(200):
            if epoch % 20 == 0:
                print ('Now training epoch %s'%epoch)
            model.train(self.taggeddoc,total_examples=model.corpus_count,epochs=model.iter)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        # shows the similar words
        print (model.most_similar('suppli'))

        # shows the learnt embedding
        print (model['suppli'])


if __name__ == '__main__':
    esSimilarity = ElasticSimilarity()
    esSimilarity.train(START_DOC, TRAIN_DOCS)
