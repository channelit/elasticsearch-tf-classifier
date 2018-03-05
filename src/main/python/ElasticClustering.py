import nltk, math, codecs
import gensim
import os
import re
from elasticsearch import Elasticsearch, helpers
from nltk.cluster.kmeans import KMeansClusterer

import string
from TextCleaner import TextCleaner

text_cleaner = TextCleaner()
from _config import ConfigMap

NUM_CLUSTERS = 5

es = ConfigMap("ElasticSearch")
training = ConfigMap("Training")
TRAIN_DOCS = int(training['size'])

class ElasticClustering:

    def __init__(self):
        self.model_file = os.path.join(training['basedir'], 'doc_model')
        self.es = Elasticsearch([es['server']], port=es['port'], http_auth=(es['user'], es['secret']))
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.model_file + '.word2vec')
        self.model = gensim.models.Doc2Vec.load(self.model_file)

    def get_titles_by_cluster(self, id):
        list = []
        for x in range(0, len(assigned_clusters)):
            if (assigned_clusters[x] == id):
                list.append(used_lines[x])
        return list

    def get_topics(self, titles):
        from collections import Counter
        words = [preprocess_document(x) for x in titles]
        words = [word for sublist in words for word in sublist]
        filtered_words = [word for word in words if word not in stopwords.words('english')]
        count = Counter(filtered_words)
        print(count.most_common()[:5])

    def cluster_to_topics(self, id):
        get_topics(get_titles_by_cluster(id))

    def es_docs(self):
        res = helpers.scan(index=es['index'], size=TRAIN_DOCS, scroll='1m',
                           client=self.es, preserve_order=True,
                           query={"query": {"match_all": {}}},
                           )
        ctr = 0
        for hit in res:
            if ctr % 50 == 0:
                print("ctr =", ctr)
            if ctr > TRAIN_DOCS:
                raise StopIteration
            if es['textfield'] in hit["_source"]:
                # print("%(category)s %(text)s" % hit["_source"])
                text = eval(es['textfieldobj'])
                text = text.replace('\\n', ' ')
                id = hit["_id"]
                yield text, id

    def cluster_docs(self):

        vectors = []
        used_lines = []

        for doc, id in self.es_docs():
            tokens = text_cleaner.clean_tokens(doc)
            if tokens != 'NC' and len(tokens) > 200:
                used_lines.append(tokens)
                vectors.append(self.model.infer_vector(tokens))

        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
        assigned_clusters = kclusterer.cluster(vectors, assign_clusters=True)

        print("done")


if __name__ == '__main__':
    esClustering = ElasticClustering()
    esClustering.cluster_docs()
