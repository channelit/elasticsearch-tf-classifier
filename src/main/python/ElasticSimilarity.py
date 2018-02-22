import gensim
import nltk
import os
from elasticsearch import Elasticsearch, helpers

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
BATCH_SIZE = 15
query=ConfigMap("QueryTypes")


class ElasticSimilarity:

    def __init__(self):
        self.model_file = os.path.join(training['basedir'], 'doc_model')
        self.es = Elasticsearch([es['server']], port=es['port'])
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.model_file + '.word2vec')
        self.model = gensim.models.Doc2Vec.load(self.model_file)

    def es_doc(self, doc_id):
        res = self.es.get(index=es['index'], http_auth=(es['user'], es['secret']), id=doc_id, doc_type=es['type'])
        text = eval(es['textfieldobj'])
        text = text.replace('\\n', ' ')
        return text

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

    def es_docs(self):
        query_match_all = {"query": {"match_all": {}}}
        query_no_similar = {
            "query" : {
                "bool" : {
                    "must_not" : [{
                        "exists" : {
                            "field" : "similar_docs,error"
                        }
                    }]
                }
            }
        }
        res = helpers.scan(index=es['index'], size=BATCH_SIZE, scroll='1m', client=self.es, preserve_order=True,
                           query=eval(query['similarity']))

        res = list(res)
        for hit in res:
            if es['textfield'] in hit["_source"]:
                text = eval(es['textfieldobj'])
                text = text.replace('\\n', ' ').replace('\\t', ' ')
                text = ''.join([x if x.isalpha() or x.isspace() else " " for x in text])
                text = text.strip()
                id = hit["_id"]
                if len(text) > 100:
                    yield text, id

    def similar(self, doc_id):
        doc = self.es_doc(doc_id)
        tokens = self.clean_tokens(doc)
        infer = self.model.infer_vector(tokens)
        similar = self.model.docvecs.most_similar([infer])
        print(similar)

    def populate_similars(self):
        for text, id in self.es_docs():
            print("processing doc id:", id)
            try:
                tokens = self.clean_tokens(text)
                infer = self.model.infer_vector(tokens)
                similar = self.model.docvecs.most_similar([infer])
                similar_ids = [s[0] for s in similar]
                body = {
                    "doc": {
                        "similar_docs": similar_ids
                    }
                }
                update_response = self.es.update(index=es['index'], doc_type=es['type'], body=body, id=id, _source=False,refresh=True)
                print(update_response)
            except:
                print("error")

if __name__ == '__main__':
    esSimilarity = ElasticSimilarity()
    esSimilarity.populate_similars()