import math
import pickle
import numpy as np
import pandas as pd
import os,sys
import nltk
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords   
from nltk.probability import FreqDist



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

def clean_all(genre):

    txt_path = open("/home/balint/Asztal/onlab/genres/clean/" + genre + '.txt')

    ids = txt_path.read()
    words = word_tokenize(ids)
    
    for i in words:
            try:
                filename = open("/home/balint/Asztal/onlab/gutenberg/data/text/" + i + "_text.txt", "r")
            except:
                print("torolve: "+ i)
                words.remove(i)


    with open('/home/balint/Asztal/onlab/genres/clean/' + genre + '.txt', 'w') as f:
        for item in words:
            f.write("%s\n" % item)
                
def create_tf_matrix():

    all_book = open("/home/balint/Asztal/onlab/genres/clean/full")
    book_ids = all_book.read()

    ids = word_tokenize(book_ids)
    testset = len(ids) #* 4 // 5
    print(testset)
    
    stop_words = stopwords.words("english")

    clean_words = {}
    tf_matrix = {}

    for i in range(10):
        #lista.append(words[i])
        fname_read = '%s_counts.txt'%(ids[i])

        filename = os.path.join("/home/balint/Asztal/onlab/gutenberg/data/counts/" + fname_read)
                
        doclen = len(open(filename).readlines())

        with open(filename) as myfile:
            head = [next(myfile) for x in range(doclen // 10)]
        tf_table = {}       
        text = listToString(head)
        toked_words = word_tokenize(text)
        toke_dic = Convert(toked_words) 
        count_words = 0

        for w,n in toke_dic.items():
            if w not in stop_words:
                count_words = count_words + int(n)

        for w,n in toke_dic.items():
            if w not in stop_words:
                tf_table[w] = int(n) / count_words

        tf_matrix[i] = tf_table

    return tf_matrix

def numOfWordsInDocs(word):
    booklist_file = open("/home/balint/Asztal/onlab/genres/clean/full")
    reader = booklist_file.read()
    ids = word_tokenize(reader)

    stopWords = set(stopwords.words("english"))

    counter = 0
    for i in ids:
        file = open("/home/balint/Asztal/onlab/gutenberg/data/counts/" + i + "_counts.txt", "r")
        for line in file:
            for part in line.split():
                if word == part:
                    counter = counter + 1
                    print(part)
    print(counter)
    return (counter)
        
def create_documents_per_words():
    
    all_book = open("/home/balint/Asztal/onlab/genres/clean/full")
    book_ids = all_book.read()

    ids = word_tokenize(book_ids)
    testset = len(ids) #* 4 // 5
    print(testset)
    
    stop_words = stopwords.words("english")

    word_per_doc_table = {}

    for i in range(10):
        fname_read = '%s_counts.txt'%(ids[i])
        filename = os.path.join("/home/balint/Asztal/onlab/gutenberg/data/counts/" + fname_read)   
        doclen = len(open(filename).readlines())
        with open(filename) as myfile:
            head = [next(myfile) for x in range(doclen // 10)]

        text = listToString(head)
        toked_words = word_tokenize(text)
        toke_dic = Convert(toked_words) 
        count_words = 0

        for w,n in toke_dic.items():
            if w not in stop_words:
                if w in word_per_doc_table:
                    word_per_doc_table[w] += 1
                else:
                    word_per_doc_table[w] = 1

    return word_per_doc_table

def create_idf_matrix( count_doc_per_words, total_documents):
    all_book = open("/home/balint/Asztal/onlab/genres/clean/full")
    book_ids = all_book.read()

    ids = word_tokenize(book_ids)
    testset = len(ids) #* 4 // 5
    print(testset)
    
    stop_words = stopwords.words("english")

    idf_matrix = {}

    for i in range(10):
        fname_read = '%s_counts.txt'%(ids[i])
        filename = os.path.join("/home/balint/Asztal/onlab/gutenberg/data/counts/" + fname_read)   
        doclen = len(open(filename).readlines())
        with open(filename) as myfile:
            head = [next(myfile) for x in range(doclen // 10)]

        text = listToString(head)
        toked_words = word_tokenize(text)
        toke_dic = Convert(toked_words) 
        idf_table = {}

        for w,n in toke_dic.items():
            if w not in stop_words:
                idf_table[w] = math.log10( total_documents / float(count_doc_per_words[w]))
    
    
        idf_matrix[ids[i]] = idf_table

    return idf_matrix

def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

#print(create_tf_matrix())
#print(create_documents_per_words())
#print(create_idf_matrix(create_documents_per_words(), 10))
#print(create_tf_idf_matrix(create_tf_matrix(), create_idf_matrix(create_documents_per_words(), 10)))
clean_all("western")