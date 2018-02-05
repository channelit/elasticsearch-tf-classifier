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


class ElasticSummarizer:

    def __init__(self):
        self.model_file = os.path.join(training['basedir'], 'doc_model')
        self.es = Elasticsearch([es['server']], port=es['port'])
        self.taggeddoc = []

    def es_docs(self):
        query_match_all = {"query": {"match_all": {}}}
        query_no_summary = {
            "query" : {
                "bool" : {
                    "must_not" : {
                        "exists" : {
                            "field" : "text_summary"
                        }
                    }
                }
            }
        }
        res = helpers.scan(index=es['index'], size=BATCH_SIZE, scroll='1m', client=self.es, preserve_order=True,
                           query=query_no_summary,
                           )
        res = list(res)
        for hit in res:
            if "text" in hit["_source"]:
                # print("%(category)s %(text)s" % hit["_source"])
                text = hit["_source"][es['textfield']][0]
                text = text.replace('\\n', ' ')
                id = hit["_id"]
                yield text, id

    def clean_tokens(self, tokens):
        tokens = [w.lower() for w in tokens]
        tokens = [t for t in tokens if t.isalpha() and t not in string.punctuation]
        tokens = [porter.stem(t) for t in tokens]
        return numpy.unique(tokens)

    def populate_summaries(self):
        for text, id in self.es_docs():
            text_summary = summarize(text, ratio=0.01)
            text_keywords = keywords(text, ratio=0.01)
            text_keywords = self.clean_tokens(text_keywords.splitlines())
            body = {
                "doc": {
                    "text_summary": text_summary,
                    "text_keywords": text_keywords.tolist()
                }
            }
            update_response = self.es.update(index=es['index'], doc_type=es['type'], body=body, id=id, _source=False,
                                             refresh=True)
            print(update_response)


if __name__ == '__main__':
    esSummarizer = ElasticSummarizer()
    esSummarizer.populate_summaries()
