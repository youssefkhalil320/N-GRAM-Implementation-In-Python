from collections import defaultdict
from modules.text_processor import TextProcessor


class NgramModel(object):

    def __init__(self, n):
        self.n = n
        self.context = defaultdict(list)
        self.ngram_counter = defaultdict(float)
        self.context_counter = defaultdict(float)
        self.text_processor = TextProcessor()

    def update(self, sentence: str) -> None:
        """
        Updates Language Model
        :param sentence: input text
        """
        n = self.n
        ngrams = self.text_processor.get_ngrams(
            n, self.text_processor.tokenize(sentence))
        for ngram in ngrams:
            self.ngram_counter[ngram] += 1.0

            prev_words, target_word = ngram
            self.context[prev_words].append(target_word)
            self.context_counter[prev_words] += 1.0
