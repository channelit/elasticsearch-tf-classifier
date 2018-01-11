from elasticsearch import Elasticsearch
es = Elasticsearch(["elasticsearch"], maxsize=25)

res = es.search(index="intelligence", body={"query": {"match_all": {}}})
print("Got %d Hits:" % res['hits']['total'])
for hit in res['hits']['hits']:
    print("%(category)s %(text)s" % hit["_source"])
