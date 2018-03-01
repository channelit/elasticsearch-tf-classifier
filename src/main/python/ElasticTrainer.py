from elasticsearch import Elasticsearch, helpers
from _config import ConfigMap

import os
from gensim.models.doc2vec import TaggedDocument
from TextCleaner import TextCleaner

import gensim
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

es = ConfigMap("ElasticSearch")
training = ConfigMap("Training")
TRAIN_DOCS = int(training['size'])
text_cleaner = TextCleaner()


class ElasticTrainer:

    def __init__(self):
        self.model_file = os.path.join(training['basedir'], 'doc_model')
        self.es = Elasticsearch([es['server']], port=es['port'], http_auth=(es['user'], es['secret']))
        self.taggeddoc = []
        self.unigram_sentences_filepath = os.path.join(training['basedir'], 'unigram_sentences_all.txt')
        self.bigram_model_filepath = os.path.join(training['basedir'], 'bigram_model_all')
        self.bigram_sentences_filepath = os.path.join(training['basedir'], 'bigram_sentences_all.txt')
        self.trigram_model_filepath = os.path.join(training['basedir'], 'trigram_model_all')
        self.trigram_sentences_filepath = os.path.join(training['basedir'], 'trigram_sentences_all.txt')
        self.trigram_dictionary_filepath = os.path.join(training['basedir'], 'trigram_dict_all.dict')
        self.trigram_sentences_filepath = os.path.join(training['basedir'], 'trigram_sentences_all.txt')
        self.trigram_bow_filepath = os.path.join(training['basedir'], 'trigram_bow_corpus_all.mm')
        self.lda_model_filepath = os.path.join(training['basedir'], 'lda_model_all')
        self.LDAvis_data_filepath = os.path.join(training['basedir'], 'ldavis_prepared')
        self.LDAvis_html_filepath = os.path.join(training['basedir'], 'ldavis.html')

    def es_docs(self):
        ctr = 0
        res = helpers.scan(index=es['index'], size=5, scroll='1m', client=self.es, preserve_order=True,
                           query={"query": {"match_all": {}}})
        for hit in res:
            if es['textfield'] in hit["_source"]:
                ctr += 1
                if ctr % 50 == 0:
                    print("ctr =", ctr)
                if ctr > TRAIN_DOCS:
                    raise StopIteration
                # print("%(category)s %(text)s" % hit["_source"])
                text = eval(es['textfieldobj'])
                text = text.replace('-\\n', '')
                text = text.replace('\\n', ' ')
                text = text.replace('\n', ' ')
                id = hit["_id"]
                yield text, id

    def train_with_tokens(self):
        for doc, id in self.es_docs():
            # tokens = self.clean_tokens(doc)
            tokens = text_cleaner.clean_tokens(doc)
            if tokens != 'NC' and len(tokens) > 200:
                td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(tokens))).split(), [id])
                self.taggeddoc.append(td)
        print ('Data Loading finished')
        print (len(self.taggeddoc), type(self.taggeddoc))
        model = gensim.models.Doc2Vec(self.taggeddoc, dm=0, iter=1, window=15, seed=1337, min_count=5, workers=4,
                                      alpha=0.025, size=200, min_alpha=0.025)
        for epoch in range(200):
            if epoch % 20 == 0:
                print ('Now training epoch %s' % epoch)
            model.train(self.taggeddoc, total_examples=model.corpus_count, epochs=model.iter)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.save(self.model_file)
        model.save_word2vec_format(self.model_file + '.word2vec')

    def train_with_trigrams(self):
        trigram_model = Phrases.load(self.trigram_model_filepath)
        bigram_model = Phrases.load(self.bigram_model_filepath)
        for doc, id in self.es_docs():
            unigrams = text_cleaner.clean_tokens(doc)
            bigrams = bigram_model[unigrams]
            trigrams = trigram_model[bigrams]
            trigrams = text_cleaner.filter_terms(trigrams)
            td = TaggedDocument(trigrams, [id])
            self.taggeddoc.append(td)
        print ('Data Loading finished')
        print (len(self.taggeddoc), type(self.taggeddoc))
        model = gensim.models.Doc2Vec(self.taggeddoc, dm=0, iter=1, window=15, seed=1337, min_count=5, workers=4,
                                      alpha=0.025, size=200, min_alpha=0.025)
        for epoch in range(200):
            if epoch % 20 == 0:
                print ('Now training epoch %s' % epoch)
            model.train(self.taggeddoc, total_examples=model.corpus_count, epochs=model.iter)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.save(self.model_file)
        model.save_word2vec_format(self.model_file + '.word2vec')

    def save_sentences(self):
        f = open(self.unigram_sentences_filepath, 'w')
        for doc, id in self.es_docs():
            for sentence in text_cleaner.clean_sentences(doc):
                f.write(sentence + '\n')
        f.close()

    def save_sentences_trigram(self):
        f = open(self.trigram_sentences_filepath, 'w')
        trigram_model = Phrases.load(self.trigram_model_filepath)
        bigram_model = Phrases.load(self.bigram_model_filepath)
        for doc, id in self.es_docs():
            unigrams = text_cleaner.clean_tokens(doc)
            bigrams = bigram_model[unigrams]
            trigrams = trigram_model[bigrams]
            trigrams = text_cleaner.filter_terms(trigrams)
            trigrams = u' '.join(trigrams)
            f.write(trigrams + '\n')

    def generate_bigrams_trigrams(self):
        unigram_sentences = LineSentence(self.unigram_sentences_filepath)
        bigram_model = Phrases(unigram_sentences)
        bigram_model.save(self.bigram_model_filepath)
        f = open(self.bigram_sentences_filepath, 'w')
        for unigram_sentence in unigram_sentences:
            bigram_sentence = u' '.join(bigram_model[unigram_sentence])
            f.write(bigram_sentence + '\n')
        f.close()
        bigram_sentences = LineSentence(self.bigram_sentences_filepath)
        trigram_model = Phrases(bigram_sentences)
        trigram_model.save(self.trigram_model_filepath)
        f = open(self.trigram_sentences_filepath, 'w')
        for bigram_sentence in bigram_sentences:
            trigram_sentence = u' '.join(trigram_model[bigram_sentence])
            f.write(trigram_sentence + '\n')

    def generate_lda_topics(self):
        from gensim.corpora import Dictionary, MmCorpus
        from gensim.models.ldamulticore import LdaMulticore
        import pyLDAvis
        import pyLDAvis.gensim
        import warnings
        import _pickle as pickle

        trigram_sentences = LineSentence(self.trigram_sentences_filepath)
        trigram_dictionary = Dictionary(trigram_sentences)
        # trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)
        trigram_dictionary.compactify()
        trigram_dictionary.save(self.trigram_dictionary_filepath)

        def trigram_bow_generator(filepath):
            for sentence in LineSentence(filepath):
                yield trigram_dictionary.doc2bow(sentence)

        MmCorpus.serialize(self.trigram_bow_filepath, trigram_bow_generator(self.trigram_sentences_filepath))
        trigram_bow_corpus = MmCorpus(self.trigram_bow_filepath)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lda = LdaMulticore(trigram_bow_corpus,
                               num_topics=3,
                               id2word=trigram_dictionary,
                               workers=3)
            lda.save(self.lda_model_filepath)
        lda = LdaMulticore.load(self.lda_model_filepath)
        lda.show_topic(0)
        lda.show_topic(1)
        lda.show_topic(2)
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda, trigram_bow_corpus, trigram_dictionary)
        pyLDAvis.save_html(LDAvis_prepared, self.LDAvis_html_filepath)


if __name__ == '__main__':
    esTrainer = ElasticTrainer()
    # esTrainer.train_with_tokens()
    # esTrainer.save_sentences()
    # esTrainer.generate_bigrams_trigrams()
    # esTrainer.save_sentences_trigram()
    # esTrainer.generate_lda_topics()
    esTrainer.train_with_trigrams()
