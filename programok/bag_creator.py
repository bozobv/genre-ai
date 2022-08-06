import numpy as np
import pandas as pd
import os,sys
import random
import nltk
import csv
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt 
from itertools import islice


def readGenres():
    all_book = open("/home/balint/Asztal/onlab/genres/bags_1000_words/genre_names")
    ingenres = all_book.read()
    return word_tokenize(ingenres)

def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct

def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele  
    
    # return string  
    return str1 

def create_testset(genre, numero):
    text_file_genre = open("/home/balint/Asztal/onlab/genres/clean/" + genre +".txt")
    text_genre_ids = text_file_genre.read()

    words = word_tokenize(text_genre_ids)
    start = 0
    testset = random.sample(words, len(words) * 4 // 5)

    fil = open("/home/balint/Asztal/onlab/genres/bags/testsets/" + genre + "_" + str(numero) + "_testset.txt", "w")
    with fil as f:
        for item in testset:
            f.write("%s\n" % item)

    return testset

def create_word_bag(genre, numero):
    topwrods_number = 2000

    testset = create_testset(genre, numero)
    
    stop_words = stopwords.words("english")

    clean_words = {}
    print("start: " + genre)


    for id in testset:
        fname_read = '%s_counts.txt'%(id)
        print("fname_read:  " + fname_read )
        try:
            filename = os.path.join("/home/balint/Asztal/onlab/gutenberg/data/counts/" + fname_read)
            
            count = len(open(filename).readlines(  ))

            with open(filename) as myfile:
                head = [next(myfile) for x in range(count // 2)]
            
            #text_file = open(filename) 
            #text = text_file.read()

            text = listToString(head)

            toked_words = word_tokenize(text)
            
            toke_dic = Convert(toked_words) 

            benne = False
            
            #for w,n in filter(lambda sub: int(sub[1]) >= 0 and int(sub[1]) <= topwrods_number, toke_dic.items()):
            for w,n in toke_dic.items():
                if w not in stop_words:
                    for w2,n2 in clean_words.items():
                        if w == w2:
                            benne = True
                            n3 = int(n) + int(n2)
                            clean_words[w] =  n3
                    if benne == False:
                        clean_words[w] = int(n)


        except:
            print(fname_read + " hiba")
    
    #fout = open("/home/balint/Asztal/onlab/genres/bags/" + genre + ".txt") 
    fout = csv.writer(open("/home/balint/Asztal/onlab/genres/bags/auto/" + genre + "_" + str(numero) + ".txt", 'w'))
    #fout_all = csv.writer(open("/home/balint/Asztal/onlab/genres/bags/all.txt", 'a'))

    out = dict(sorted(clean_words.items(), key=lambda x: x[1], reverse=True))

    firsts = {k: out[k] for k in list(out)[:topwrods_number]}
    #firsts_all = {k: out[k] for k in list(out)[:topwrods_number//10]}

    #for key, val in firsts.items() :
     #   fout.writerow([key])
        
    for key, val in firsts.items() :   
        fout.writerow([key, val])


    print("end: " + genre)


genres = readGenres()

for i in range(16):
    print(i, ".  iteráció")
    for genre in genres:
        create_word_bag(genre, i)





