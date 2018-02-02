import gensim
import nltk
import os
from elasticsearch import Elasticsearch

from _config import ConfigMap

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
        self.es = Elasticsearch([es['server']], port = es['port'])
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.model_file + '.word2vec')
        self.model = gensim.models.Doc2Vec.load(self.model_file)

    def es_doc(self, doc_id):
        res = self.es.get(index=es['index'], id=doc_id, doc_type=es['type'])
        text = res["_source"][es['textfield']][0]
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

    def similar(self, doc_id):
        doc = self.es_doc(doc_id)
        tokens = self.clean_tokens(doc)
        infer = self.model.infer_vector(tokens)
        similar = self.model.docvecs.most_similar([infer])
        print(similar)

if __name__ == '__main__':
    esSimilarity = ElasticSimilarity()
    esSimilarity.similar('k1GX5WAB21MqlvbL0OFV')
