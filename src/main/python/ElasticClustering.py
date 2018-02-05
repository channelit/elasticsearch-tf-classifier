import gensim
import nltk
import os
import re
from elasticsearch import Elasticsearch, helpers
from nltk.cluster.kmeans import KMeansClusterer

nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

from _config import ConfigMap

NUM_CLUSTERS = 5

TRAIN_DOCS = 15
es = ConfigMap("ElasticSearch")
training = ConfigMap("Training")


class ElasticClustering:

    def __init__(self):
        self.model_file = os.path.join(training['basedir'], 'doc_model')
        self.es = Elasticsearch([es['server']], port=es['port'])
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.model_file + '.word2vec')
        self.model = gensim.models.Doc2Vec.load(self.model_file)

    # clustersizes = []
    #
    # def distanceToCentroid():
    #     for i in range(0,NUM_CLUSTERS):
    #         clustersize = 0
    #         for j in range(0,len(assigned_clusters)):
    #             if (assigned_clusters[j] == i):
    #                 clustersize+=1
    #         clustersizes.append(clustersize)
    #         dist = 0.0
    #         centr = means[i]
    #         for j in range(0,len(assigned_clusters)):
    #             if (assigned_clusters[j] == i):
    #                 dist += pow(nltk.cluster.util.cosine_distance(vectors[j], centr),2)/clustersize
    #         dist = math.sqrt(dist)
    #         print("distance cluster: "+str(i)+" RMSE: "+str(dist)+" clustersize: "+str(clustersize))

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
        res = helpers.scan(index=es['index'], size=TRAIN_DOCS, scroll='1m', client=self.es, preserve_order=True,
                           query={"query": {"match_all": {}}},
                           )
        res = list(res)
        for hit in res:
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
            tokens = self.clean_tokens(doc)
            if tokens != 'NC' and len(tokens) > 200:
                used_lines.append(tokens)
                vectors.append(self.model.infer_vector(tokens))

        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
        assigned_clusters = kclusterer.cluster(vectors, assign_clusters=True)

        print("done")


if __name__ == '__main__':
    esClustering = ElasticClustering()
    esClustering.cluster_docs()
