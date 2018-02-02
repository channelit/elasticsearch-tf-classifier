from elasticsearch import Elasticsearch, helpers
from _config import ConfigMap
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

TRAIN_DOCS = 5
es = ConfigMap("ElasticSearch")

class ElasticSimilarity:

    def __init__(self):

        self.base_dir = '/assets'
        self.model_file = os.path.join(self.base_dir, 'doc_model')

        self.es = Elasticsearch([es['server']], port = es['port'])

        self.taggeddoc = []

    def es_docs(self, train_docs):
        res = helpers.scan(index="intelligence", size=train_docs, scroll='1m', client = self.es, preserve_order=True,
                           query={"query": {"match_all": {}}},
                           )
        for hit in res:
            if "text" in hit["_source"] :
                # print("%(category)s %(text)s" % hit["_source"])
                text = hit["_source"]["text"][0]
                text = text.replace('\\n', ' ')
                id = hit["_id"]
                yield text, id


    def es_doc(self, doc_id):
        res = self.es.get(index="intelligence", id=doc_id, doc_type='discoverer')
        text = res["_source"]["text"][0]
        text = text.replace('\\n', ' ')
        return text

    def clean_tokens(self, text):
        try:
            tokens = word_tokenize(text)
            tokens = [w.lower() for w in tokens]
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            words = [word for word in stripped if word.isalpha()]
            words = [w for w in words if (len(w) in range(2,12) and not w in stop_words)]
            words = [porter.stem(w) for w in words]
            return words
        except:
            return 'NC'

    def train(self, train_docs):
        for doc, id in self.es_docs(train_docs):
            tokens = self.clean_tokens(doc)
            if tokens != 'NC':
                td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(tokens))).split(),[id])
                self.taggeddoc.append(td)

        print ('Data Loading finished')
        print (len(self.taggeddoc),type(self.taggeddoc))

        model = gensim.models.Doc2Vec(self.taggeddoc, dm = 0, alpha=0.025, size= 20, min_alpha=0.025, min_count=0)

        for epoch in range(200):
            if epoch % 20 == 0:
                print ('Now training epoch %s'%epoch)
            model.train(self.taggeddoc,total_examples=model.corpus_count,epochs=model.iter)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.save(self.model_file)
        model.save_word2vec_format(self.model_file + '.word2vec')

    def similar(self, doc_id):
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.model_file + '.word2vec')
        model = gensim.models.Doc2Vec.load(self.model_file)
        doc = self.es_doc(doc_id)
        tokens = self.clean_tokens(doc)
        infer = model.infer_vector(tokens)
        similar = model.docvecs.most_similar([infer])
        print(similar)

if __name__ == '__main__':
    esSimilarity = ElasticSimilarity()
    esSimilarity.train(TRAIN_DOCS)
    esSimilarity.similar('k1GX5WAB21MqlvbL0OFV')
