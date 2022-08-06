import numpy as np
import pandas as pd
import os,sys

import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt 
import csv

path_gutenberg = os.path.join(os.pardir,os.pardir,'gutenberg')
src_dir = '/home/balint/Asztal/onlab/gutenberg-analysis/src'
sys.path.append(src_dir)
src_dir = os.path.join(os.pardir,'src')
sys.path.append(src_dir)
from data_io import get_book

path_read = os.path.join(path_gutenberg,'data','counts')

def create_word_bag(genre):

    text_file_genre = open("/home/balint/Asztal/onlab/genres/clean/" + genre +".txt")
    text_genre_ids = text_file_genre.read()

    words = word_tokenize(text_genre_ids)
    testset = len(words) * 4 // 5
    print(testset)

    stop_words = stopwords.words("english")

    clean_words = []
    
    print("start: " + genre)


    for i in range(testset):
        #lista.append(words[i])
        fname_read = '%s_counts.txt'%(words[i])
        #print("fname_read:  " + fname_read )
        try:
            filename = os.path.join("/home/balint/Asztal/onlab/gutenberg/data/counts/" + fname_read)
            #print("filename:  " + filename )

            text_file = open(filename) 
            text = text_file.read()

            toked_words = word_tokenize(text)
            #print("toked_words:  " + len(toked_words) )

            words_no_punc = []

            for w in toked_words:
                if w.isalpha():
                    words_no_punc.append(w.lower())

            #print("words_no_punc:  " + len(words_no_punc) )

            counter = 0

            for w in words_no_punc:
                if w not in stop_words:
                    clean_words.append(w)
                    counter +=1
                if counter > 200:
                    break

            #print("clean_words:  " + len(clean_words) )

        except:
            print(fname_read + " nope")
    
    print(len(clean_words))

    fdist = FreqDist(clean_words)
    #fout = open("/home/balint/Asztal/onlab/genres/bags/" + genre + ".txt") 
    
    fout = csv.writer(open("/home/balint/Asztal/onlab/genres/bags/" + genre + ".txt", 'w'))
    for key,n in fdist.most_common(300):
        fout.writerow([key])
    print("end: " + genre)









#print(len(clean_words))

create_word_bag("western")
#create_word_bag("fantasy")
#create_word_bag("scifi")
#create_word_bag("romance")
#create_word_bag("detective")

    


#for i in range(testset, len(words)):
#    print(words[i])




#level = 'counts'
#dict_word_count = get_book(pg_id, level=level)
#print(dict_word_count)

