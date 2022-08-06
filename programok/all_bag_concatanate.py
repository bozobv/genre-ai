import os,sys

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt 


src_dir = '/home/balint/Asztal/onlab/genres/bags/'
sys.path.append(src_dir)

filenames = ['western.txt', 'fantasy.txt', 'detective.txt' , 'scifi.txt', 'romance.txt']
with open('/home/balint/Asztal/onlab/genres/bags/all.txt', 'w') as outfile:
    for fname in filenames:
        with open('/home/balint/Asztal/onlab/genres/bags/' + fname) as infile:
            outfile.write(infile.read())


text_file = open('/home/balint/Asztal/onlab/genres/bags/all.txt')
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


mylist = list(dict.fromkeys(clean_words))

with open('/home/balint/Asztal/onlab/genres/bags/all.txt', 'w') as filehandle:
    for listitem in mylist:
        filehandle.write('%s\n' % listitem)