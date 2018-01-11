from elasticsearch import Elasticsearch

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

es = Elasticsearch(["elasticsearch"], maxsize=25)


res = es.search(index="intelligence", body={"query": {"match_all": {}}})
print("Got %d Hits:" % res['hits']['total'])
for hit in res['hits']['hits']:
    # print("%(category)s %(text)s" % hit["_source"])
    text = hit["_source"]["text"][0]
    words = word_tokenize(text)
    print(words)
