
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
        return ''.join([i if ord(i) < 128 else ' ' for i in text])

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

    def clean_tokens(self, text):
        nlp_text = self.nlp_text(text)
        tokens = [w for w in nlp_text if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num]
        clean_tokens = [self.cleanup(t.string) for t in tokens if not self.isNoise(t)]
        return clean_tokens

if __name__ == "__main__":
    print("In Text Cleaner")