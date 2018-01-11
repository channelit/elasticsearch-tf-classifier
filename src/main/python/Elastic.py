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


res = es.search(index="intelligence", body={"query": {"match_all": {}}})
print("Got %d Hits:" % res['hits']['total'])
for hit in res['hits']['hits']:
    # print("%(category)s %(text)s" % hit["_source"])
    text = hit["_source"]["text"][0]
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in tokens if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    print(words)
