# Project 2 - Chatbot Project
# nlp_utils file - contains logic for stemming, tokenization, and applying bag of words technique
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer


# used to convert phrases to a vector of numbers in a common vector space to be able to calculate similarities
def make_word_bag(sentence_tokens, total_words):
    sentence_tokens = [stem(token) for token in sentence_tokens]  # gets tokens from sentence token list
    word_bag = np.zeros(len(total_words), dtype=np.float)
    for index, word in enumerate(total_words):
        if word in sentence_tokens:
            # Set value for current column in vector to 1 if it matches element in total vocab
            word_bag[index] = 1.0
    return word_bag


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    stem_machine = PorterStemmer()
    word_stem = stem_machine.stem(word.lower())
    return word_stem
