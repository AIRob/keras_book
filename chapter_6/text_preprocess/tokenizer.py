from keras.preprocessing.text import Tokenizer, one_hot , text_to_word_sequence
import numpy as np


def test_text_to_word_sequence():
    sequence = text_to_word_sequence('The cat sat on the mat. \
                                      The dog sat on the log. \
                                      Dogs and cats living together.')
    print(sequence)

def test_one_hot():
    text = 'The cat sat on the mat.'
    encoded = one_hot(text, 5)
    print(encoded)

def test_tokenizer():
    texts = ['The cat sat on the mat.',
             'The dog sat on the log.',
             'Dogs and cats living together.']
    tokenizer = Tokenizer(num_words=10)
    tokenizer.fit_on_texts(texts)
    print('word_counts: ', tokenizer.word_counts)
    print('word_docs: ', tokenizer.word_docs)
    print('word_index: ', tokenizer.word_index)
    print('document_count: ', tokenizer.document_count)

test_text_to_word_sequence()
test_one_hot()
test_tokenizer()
