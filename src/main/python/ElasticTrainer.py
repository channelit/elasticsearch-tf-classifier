from elasticsearch import Elasticsearch, helpers
from _config import ConfigMap
import nltk
import gensim
import os
from gensim.models.doc2vec import TaggedDocument

nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

TRAIN_DOCS = 15
es = ConfigMap("ElasticSearch")
training = ConfigMap("Training")


class ElasticSimilarity:

    def __init__(self):
        self.model_file = os.path.join(training['basedir'], 'doc_model')
        self.es = Elasticsearch([es['server']], port=es['port'])
        self.taggeddoc = []

    def es_docs(self):
        res = helpers.scan(index=es['index'], size=TRAIN_DOCS, scroll='1m', client=self.es, preserve_order=True,
                           query={"query": {"match_all": {}}},
                           )
        res = list(res)
        for hit in res:
            if "text" in hit["_source"]:
                # print("%(category)s %(text)s" % hit["_source"])
                text = eval(es['textfieldobj'])
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
            words = [w for w in words if (len(w) in range(2, 12) and not w in stop_words)]
            words = [porter.stem(w) for w in words]
            return words
        except:
            return 'NC'

    def train(self):
        for doc, id in self.es_docs():
            tokens = self.clean_tokens(doc)
            if tokens != 'NC' and len(tokens) > 200:
                td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(tokens))).split(), [id])
                self.taggeddoc.append(td)

        print ('Data Loading finished')
        print (len(self.taggeddoc), type(self.taggeddoc))

        model = gensim.models.Doc2Vec(self.taggeddoc, dm=0, iter=1, window=5, seed=1337, min_count=5, workers=4,
                                      alpha=0.025, size=200, min_alpha=0.025)

        for epoch in range(200):
            if epoch % 20 == 0:
                print ('Now training epoch %s' % epoch)
            model.train(self.taggeddoc, total_examples=model.corpus_count, epochs=model.iter)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.save(self.model_file)
        model.save_word2vec_format(self.model_file + '.word2vec')


if __name__ == '__main__':
    esSimilarity = ElasticSimilarity()
    esSimilarity.train()
