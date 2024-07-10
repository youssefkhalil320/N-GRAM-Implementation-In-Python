import random
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

    def prob(self, context, token):
        """
        Calculates probability of a candidate token to be generated given a context
        :return: conditional probability
        """
        count_of_token = self.ngram_counter.get((context, token), 0.0)
        count_of_context = self.context_counter.get(context, 0.0)
        if count_of_context == 0:
            return 0.0
        return count_of_token / count_of_context

    def random_token(self, context):
        """
        Given a context we "semi-randomly" select the next word to append in a sequence
        :param context:
        :return:
        """
        if context not in self.context:
            return '<END>'

        tokens = self.context[context]
        probs = [self.prob(context, token) for token in tokens]
        return random.choices(tokens, weights=probs, k=1)[0]
