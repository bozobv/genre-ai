import numpy as np
import pandas as pd
import os,sys

from collections import Counter
import matplotlib.pyplot as plt

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

path_gutenberg = '/home/balint/Asztal/onlab/gutenberg'
src_dir = '/home/balint/Asztal/onlab/gutenberg-analysis/src'
sys.path.append(src_dir)

from data_io import get_book

text_file = open("/home/balint/Asztal/onlab/gutenberg/data/counts/PG108_counts.txt")
text = text_file.read()

words = word_tokenize(text)


words_no_punc = []
clean_words = []
stopwords = stopwords.words("english")

for w in words:
    if w.isalpha():
        words_no_punc.append(w.lower())
    
for w in words_no_punc:
    if w not in stopwords:
        clean_words.append(w)

#print(words_no_punc)
#print("\n")

#print (clean_words)
lemmatizer = WordNetLemmatizer()

lemmatized_words = []

for w in clean_words:
   lemmatized_words.append(lemmatizer.lemmatize(w))

print((lemmatized_words))