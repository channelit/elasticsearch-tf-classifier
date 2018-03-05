from elasticsearch import Elasticsearch, helpers
from _config import ConfigMap
import nltk
from gensim.summarization import summarize
from gensim.summarization import keywords
import os
import numpy
import json

nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

BATCH_SIZE = 15
es = ConfigMap("ElasticSearch")
training = ConfigMap("Training")
query = ConfigMap("QueryTypes")
TRAIN_DOCS = int(training['size'])

class ElasticSummarizer:

    def __init__(self):
        self.model_file = os.path.join(training['basedir'], 'doc_model')
        self.es = Elasticsearch([es['server']], port=es['port'], http_auth=(es['user'], es['secret']))
        self.taggeddoc = []

    def es_docs(self):
        query_match_all = {"query": {"match_all": {}}}
        query_no_summary = {
            "query": {
                "bool": {
                    "must_not": [{
                        "exists": {
                            "field": "text_summary,error"
                        }
                    }]
                }
            }
        }
        res = helpers.scan(index=es['index'], size=BATCH_SIZE, scroll='1m',
                           client=self.es, preserve_order=True,
                           query=eval(query['summary']),
                           )
        ctr = 0
        for hit in res:
            ctr += 1
            if ctr % 50 == 0:
                print("ctr =", ctr)
            if ctr > TRAIN_DOCS:
                raise StopIteration
            if es['textfield'] in hit["_source"]:
                text = eval(es['textfieldobj'])
                text = text.replace('\\n', ' ').replace('\\t', ' ')
                text = ''.join([x if x.isalpha() or x.isspace() else " " for x in text])
                text = text.strip()
                id = hit["_id"]
                if len(text) > 100:
                    yield text, id

    def clean_tokens(self, tokens):
        tokens = [w.lower() for w in tokens]
        tokens = [t for t in tokens if t.isalpha() and t not in string.punctuation]
        tokens = [porter.stem(t) for t in tokens]
        return numpy.unique(tokens)

    def populate_summaries(self):
        for text, id in self.es_docs():
            print("processing doc id:", id)
            try:
                text_summary = summarize(text, ratio=0.8)
                text_keywords = keywords(text, ratio=0.8)
                text_keywords = self.clean_tokens(text_keywords.splitlines())
                body = {
                    "doc": {
                        "text_summary": text_summary,
                        "text_keywords": text_keywords.tolist()
                    }
                }
                update_response = self.es.update(index=es['index'], doc_type=es['type'], body=body, id=id,
                                                 _source=False, refresh=True)
                print(update_response)
            except:
                print("error")


if __name__ == '__main__':
    esSummarizer = ElasticSummarizer()
    esSummarizer.populate_summaries()
