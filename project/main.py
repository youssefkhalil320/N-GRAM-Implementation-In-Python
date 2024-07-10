import time
import random
from modules.text_processor import TextProcessor
from model.ngram import NgramModel
from train import create_ngram_model

start = time.time()
m = create_ngram_model(6, '../data/Frankenstein.txt')
print(f'Language Model creating time: {time.time() - start}')
start = time.time()
random.seed(7)
print(f'{"="*50}\nGenerated text:')
print(m.generate_text(20))
print(f'{"="*50}')
