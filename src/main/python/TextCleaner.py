
import gensim
import spacy
import numpy as np

class TextCleaner:

    def __init__(self):
        my_stop_words = [u'say', u'\'s', u'Mr', u'be', u'said', u'says', u'saying']
        self.nlp = spacy.load("en")

        for stopword in my_stop_words:
            lexeme = self.nlp.vocab[stopword]
            lexeme.is_stop = True

    def unicode(self, text):
        return ''.join([i if ord(i) < 128 else ' ' for i in text])

    def nlp_text(self, text):
        return self.nlp(self.unicode(text))

    def clean_tokens(self, text):
        nlp_text = self.nlp_text(text)
        tokens = [w for w in nlp_text if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num]
        bigram = gensim.models.Phrases(tokens)
        return [bigram[line] for line in tokens]

if __name__ == "__main__":
    print("In Text Cleaner")