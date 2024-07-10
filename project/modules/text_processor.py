import string
from typing import List, Tuple


class TextProcessor:
    def __init__(self):
        self.punctuation = string.punctuation

    def tokenize(self, text: str) -> List[str]:
        """
        :param text: Takes input sentence
        :return: tokenized sentence
        """
        for punct in self.punctuation:
            text = text.replace(punct, ' ' + punct + ' ')
        tokens = text.split()
        return tokens

    def get_ngrams(self, n: int, tokens: List[str]) -> List[Tuple[Tuple[str, ...], str]]:
        """
        :param n: n-gram size
        :param tokens: tokenized sentence
        :return: list of ngrams
        ngrams of tuple form: ((previous words), target word)
        """
        tokens = (n-1)*['<START>'] + tokens
        ngrams = [(tuple(tokens[i-p-1] for p in reversed(range(n-1))), tokens[i])
                  for i in range(n-1, len(tokens))]
        return ngrams
