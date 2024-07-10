from modules.text_processor import TextProcessor
from model.ngram import NgramModel

# text_processor = TextProcessor()
# text = "This is a sample text for n-gram generation."
# tokens = text_processor.tokenize(text)
# ngrams = text_processor.get_ngrams(3, tokens)
# print(ngrams)

model = NgramModel(3)
text = "This is a sample text for n-gram generation."
model.update(text)
