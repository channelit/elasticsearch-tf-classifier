
import spacy

noisy_pos_tags = ['PROP']
min_token_length = 5

class TextCleaner:

    def __init__(self):
        my_stop_words = [u'say', u'\'s', u'Mr', u'be', u'said', u'says', u'saying']
        self.nlp = spacy.load("en_core_web_sm")


        for stopword in my_stop_words:
            lexeme = self.nlp.vocab[stopword]
            lexeme.is_stop = True

    def unicode(self, text):
        return ''.join([i if (i.isalpha() and ord(i) < 128) or i == '\n' else ' ' for i in text])

    def nlp_text(self, text):
        return self.nlp(self.unicode(text))


    def isNoise(self, token):
        is_noise = False
        if token.pos_ in noisy_pos_tags:
            is_noise = True
        elif token.is_stop:
            is_noise = True
        elif len(token.string) <= min_token_length:
            is_noise = True
        return is_noise

    def cleanup(self, token, lower = True):
        if lower:
            token = token.lower()
        return token.strip()

    def lemmatized_sentence_corpus(self, nlp_text):
        for sent in nlp_text.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not self.punct_space(token)])

    def punct_space(self, token):
        return token.is_punct or token.is_space or token.like_num or token.is_stop or token.__len__() < 5

    def clean_tokens(self, text):
        nlp_text = self.nlp_text(text)
        tokens = [w for w in nlp_text if not w.is_stop and not w.is_punct and not w.like_num and not w.is_space]
        clean_tokens = [self.cleanup(t.string) for t in tokens if not self.isNoise(t)]
        return clean_tokens

    def clean_sentences(self, text):
        nlp_text = self.nlp_text(text)
        sentences = [sent for sent in self.lemmatized_sentence_corpus(nlp_text)]
        return sentences

    def filter_terms(self, terms):
        terms = [term for term in terms if term not in spacy.lang.en.STOP_WORDS]
        return terms

    # def clean_tokens(self, text):
    #     import nltk
    #     nltk.download('punkt')
    #     from nltk.tokenize import word_tokenize
    #     import string
    #     nltk.download('stopwords')
    #     from nltk.corpus import stopwords
    #     stop_words = set(stopwords.words('english'))
    #     from nltk.stem.porter import PorterStemmer
    #     porter = PorterStemmer()
    #     try:
    #         tokens = word_tokenize(text)
    #         tokens = [w.lower() for w in tokens]
    #         table = str.maketrans('', '', string.punctuation)
    #         stripped = [w.translate(table) for w in tokens]
    #         words = [word for word in stripped if word.isalpha()]
    #         words = [w for w in words if (len(w) in range(2, 12) and not w in stop_words)]
    #         words = [porter.stem(w) for w in words]
    #         return words
    #     except:
    #         return 'NC'

if __name__ == "__main__":
    print("In Text Cleaner")