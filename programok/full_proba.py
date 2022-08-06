import math
import pickle
import numpy as np
import pandas as pd
import os,sys
import nltk
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords   
from nltk.probability import FreqDist
import re
from scipy.spatial import distance
import collections
import operator

howmuch = 600

def ix( dic, n): 
   try:
       return list(dic)[n] # or sorted(dic)[n] if you want the keys to be sorted
   except IndexError:
       print("not enough keys")

def sort_dict(unsorted_dict):
    sorted_d = dict( sorted(unsorted_dict.items(), key=operator.itemgetter(1),reverse=True))
    return sorted_d
    
def readGenres():
    all_book = open("/home/balint/Asztal/onlab/genres/bags_1000_words/genre_names")
    ingenres = all_book.read()
    return word_tokenize(ingenres)

def getList(dict): 
    return list(dict.keys())

def get_merged_words(corrected_attrs, genres):
    merged = []
    for genre in genres:
            for w in corrected_attrs[genre]:
                if w not in merged:
                    merged.append(w)
    return merged

def merge_corrected_genre_words(attrs, corrected_attrs, merged_words, genres):
    
    mergedDict = corrected_attrs
    finalized_attrs = {}

    for genre in genres:
        
        genre_merged = mergedDict[genre]
        genre_attrs = attrs[genre]
        for w in merged_words:
            if w not in genre_merged:
                if w in genre_attrs:
                   genre_merged[w] = genre_attrs[w]
                else:
                    genre_merged[w] = 0

        finalized_attrs[genre] = genre_merged  
        #print()
        #print(genre_merged)  
    #print(finalized_attrs)
    return finalized_attrs

def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst) - 1, 2)}
    return res_dct

def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele  
    
    # return string  
    return str1 

#tf mátrix létrehozása: minden műfajban megnézi, hogy egy szó előfurdulása mennyi arányosan a többihez kéepst
def create_tf_matrix(numero):

    genres = readGenres()
    
    clean_words = {}
    tf_matrix = {}

    for genre in genres:

        fname_read = genre + "_" + str(numero) + ".txt"
        filename = os.path.join("/home/balint/Asztal/onlab/genres/bags/auto/" + fname_read) 
 
        #doclen = len(open(filename).readlines())

        with open(filename) as myfile:
            #head = [next(myfile) for x in range(doclen // howmuch)]
            head = [next(myfile) for x in range(howmuch)]


        tf_table = {}  
        text = listToString(head)        
        res = re.split(',|\n', text)

        toke_dic = Convert(res) 
        count_words = 0
        
        #összaadja a szavak előfordulás számát
        for w,n in toke_dic.items():
            count_words = count_words + int(n)

        #minden szónak megnézi hányad részét képezi az előfordulása az összeshez képest
        for w,n in toke_dic.items():
            tf_table[w] = int(n) / count_words

        tf_matrix[genre] = tf_table

    return tf_matrix

#az összes leggyakoribb szóra megnézi, hogy az összes könyv közül mennyiben fordul elő    
def create_documents_per_words(numero):
    word_per_doc_table = {}
    genres = readGenres()
    

    for genre in genres:
        fname_read = genre + "_" + str(numero) + ".txt"
        try:
            filename = os.path.join("/home/balint/Asztal/onlab/genres/bags/auto/" + fname_read) 
            doclen = len(open(filename).readlines())

            with open(filename) as myfile:
                #head = [next(myfile) for x in range(doclen//howmuch)]
                head = [next(myfile) for x in range(howmuch)]

            text = listToString(head)        
            res = re.split(',|\n', text)

            toke_dic = Convert(res) 

            for w,n in toke_dic.items():
                if w in word_per_doc_table:
                    word_per_doc_table[w] += 1
                else:
                    word_per_doc_table[w] = 1
        except:
            print("create_documents_per_words  hibás beolvasás")
    return word_per_doc_table

#műfajonként megadja, hogy az 5 műfajból melyik szó mennyire műfajspecifikus
def create_idf_matrix( count_doc_per_words, total_documents, numero):
    idf_matrix = {}

    genres = readGenres()
    
    for genre in genres:

        fname_read = genre + "_" + str(numero) + ".txt"
        filename = os.path.join("/home/balint/Asztal/onlab/genres/bags/auto/" + fname_read) 
                
        doclen = len(open(filename).readlines())

        with open(filename) as myfile:
            #head = [next(myfile) for x in range(doclen//howmuch)]
            head = [next(myfile) for x in range(howmuch)]

        tf_table = {}  
        text = listToString(head)        
        res = re.split(',|\n', text)

        toke_dic = Convert(res) 

        idf_table = {}

        for w,n in toke_dic.items():
            idf_table[w] = math.log10( total_documents / float(count_doc_per_words[w]))
    

        idf_matrix[genre] = idf_table

    return idf_matrix

#tfidf mátrix létrehozása
def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (genre1, word_dic1), (genre2, word_dic2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(word_dic1.items(), word_dic2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[genre1] = tf_idf_table

    return tf_idf_matrix

#minden műfaj szavaiba beleteszi azokat a szavakat nulla értékkel, amik a többi műfajban benne vannak
def create_attribes(idf_matrix, genres, numero):
    attribes = create_documents_per_words(numero).keys()
    
    attrib_matrix = {}
    for genre in genres:
        table = {}
        for w in attribes:
            if w in idf_matrix[genre]:
                table[w] = idf_matrix[genre][w]
            else:
                table[w] = 0 
                
        attrib_matrix[genre] = table

    return attrib_matrix

def get_book_words(book_id):
    
    filename = "/home/balint/Asztal/onlab/gutenberg/data/counts/" + book_id + "_counts.txt"
    stop_words = stopwords.words("english")

    count = len(open(filename).readlines( )) 
    with open(filename) as myfile:
        head = [next(myfile) for x in range(count // 2)]

    clean_tokes = {}

    text = listToString(head)
    toked_words = word_tokenize(text)
    toke_dic = Convert(toked_words) 

    for w,n in toke_dic.items():
        if w not in stop_words:
            clean_tokes[w] = n
    #print(clean_tokes)
    return clean_tokes
    
def get_book_words_only(book_id):
    filename = "/home/balint/Asztal/onlab/gutenberg/data/counts/" + book_id + "_counts.txt"
    stop_words = stopwords.words("english")

    count = len(open(filename).readlines( )) 
    with open(filename) as myfile:
        head = [next(myfile) for x in range(count * 2 // 3)]

    clean_tokes = {}

    text = listToString(head)
    toked_words = word_tokenize(text)
    
    return toked_words

def get_attrib_words(attrib_matrix):
    attrib_words = []
    for w in attrib_matrix["detective"]:
        attrib_words.append(w) 
    
    return attrib_words

def create_idf_for_book(count_doc_per_words, total_documents, attrib_words):
    idf_table = {}
    for w in attrib_words:
        idf_table[w] = math.log10( total_documents / float(count_doc_per_words[w]))
    
    return idf_table

def create_tf_for_book(book_id, attrib_words):
    
    book_words = get_book_words(book_id)

    count_words = 0

    tf_table = {}

    for w,n in book_words.items():
        count_words = count_words + int(n)

    for w,n in book_words.items():
        if w in attrib_words:
            tf_table[w] = int(n) / count_words
    
    return tf_table

def create_book_vector(merged_words, tf_table, idf_table):
    
    for w in merged_words:
        if w not in tf_table:
            tf_table[w] = 0


    tf_idf_table = {}

    for (word1, value1), (word2, value2) in zip(tf_table.items(), idf_table.items()):  # here, keys are the same in both the table
        tf_idf_table[word1] = float(value1 * value2)

    tf_idf_table = tf_idf_table.items()
    tf_idf_table = sorted(tf_idf_table)
    out = dict(tf_idf_table)

    return out
     
def eucl_distance(attrs, book_vector, genres):
    
    #print(book_vector)
    book_vales = book_vector.values()
    
    book_vales = list(book_vales)
    
    distances = {}
    debug= {}

    for genre in genres:
        gvec = attrs[genre].items()
        gvec = sorted(gvec)
        gvec = dict(gvec)
        debug[genre] = gvec
        gvecvals = gvec.values()
        gvecvals = list(gvecvals)
        distances[genre] = distance.euclidean(book_vales, gvecvals)
    
    

    return distances

def manhattan_distance(attrs, book_vector, genres):
    
    #print("book vector:", book_vector)
    book_vales = book_vector.values()
    #print()
    book_vales = list(book_vales)
    
    distances = {}
    debug= {}

    for genre in genres:
        #print(genre)
        gvec = attrs[genre].items()
        gvec = sorted(gvec)
        gvec = dict(gvec)
        debug[genre] = 0 #gvec
        gvec_vals = gvec.values()
        gvec_vals = list(gvec_vals)
        dist = 0
        for i in range(len(gvec_vals)):
            #print(i, "  " ,ix(gvec, i))
            #if (gvec_vals[i] != 0) and (book_vales[i] != 0):
            #print(" gvec[i]: ",gvec_vals[i])
            #print(" book[i] ",book_vales[i])
            #print(abs(gvec_vals[i] - book_vales[i]))
            dist = dist + abs(gvec_vals[i] - book_vales[i])
            debug[genre] = debug[genre] + gvec_vals[i]
            
        distances[genre] = dist
    
    #print()
    #print("scifi: ",debug["scifi"])
    #print()
    #print("detective", debug["detective"])
    #print()
    #print(debug)
    #print()
    #print(distances)

    return distances

def cosine_distance(attrs, book_vector, genres):
    book_vales = book_vector.values()
    book_vales = list(book_vales)


    distances = {}

    for genre in genres:
        #print(genre)
        gvec = attrs[genre].items()
        gvec = sorted(gvec)
        gvec = dict(gvec)
        gvec_vals = gvec.values()
        gvec_vals = list(gvec_vals)
    
        dist = distance.cosine(gvec_vals, book_vales)   
        #print()
        distances[genre] = dist

    return distances

def nearest_distance(dists):
    out = min(dists, key = dists.get)
    return out

def prediction(genres, idf_for_book, merged_words , book_id, attrs):
    try:
        book_tf = create_tf_for_book(book_id, merged_words)
        #print(book_tf)
        book_vector = create_book_vector(merged_words ,book_tf, idf_for_book )
        
        dist = cosine_distance(attrs, book_vector, genres)
        #dist = eucl_distance(attrs, book_vector, genres)
        #dist = eucl_distance(attrs, book_vector, genres)
        #print(dist)
        return nearest_distance(dist)
    except:
        #print("rossz")
        return "non"

def get_testerset(genre, numero):
    text_file_set = open("/home/balint/Asztal/onlab/genres/bags/auto/" + genre + "_" + str(numero) + ".txt")

    text_genre_set_ids = text_file_set.read()
    set_ids = word_tokenize(text_genre_set_ids)

    text_file_full = open("/home/balint/Asztal/onlab/genres/clean/" + genre +".txt")
    text_full_ids = text_file_full.read()
    full_ids = word_tokenize(text_full_ids)

    test_ids = []

    for id in full_ids:
        if id not in set_ids:
            test_ids.append(id)

    return test_ids

def test(genre, idf_for_book, attrib_words, attrs, numero):
    
    #testsetStart = 0
    testerset = get_testerset(genre, numero)

    testset = len(testerset)

    scores = ["western" , 0  , "scifi" , 0, "detective" , 0 , "romance" , 0  ,"fantasy" , 0]
    scores_dic = Convert(scores)

    

    for id in testerset:

        predict = prediction(readGenres(), idf_for_book, attrib_words, id, attrs )
        
        
        if (predict == "non"):
            testset = testset - 1
            #print("meret csokkentve")
        else:
            #if predict == genre:
                #print("helyes")
            #else:
                #print("helytelen")
            scores_dic[predict] = scores_dic[predict] + 1
    
    for i in scores_dic:
        scores_dic[i] = scores_dic[i] / testset

    #print("teszthalmaz mérete:" , testset)

    return scores_dic

def longy_test(attrib_words, book_words):
    out = {}
    for w in attrib_words:
        if w in book_words:
            out[w] = book_words[w]
    return out

def longest(words, attrs):
    out = {}
    for g in readGenres():
        asd = 0
        for w in attrs[g]:
            if w in words:
                asd = asd + float(words[w]) * attrs[g][w]
        out[g] = asd

    return max(out.items(), key=operator.itemgetter(1))[0]
    
def test2(genre, attrs, attrib_words):
    text_file_genre = open("/home/balint/Asztal/onlab/genres/clean/" + genre +".txt")

    text_genre_ids = text_file_genre.read()
    ids = word_tokenize(text_genre_ids)
    
    testsetStart = len(ids) * 4 // 5
    testset = len(ids) - testsetStart

    scores = ["western" , 0  , "scifi" , 0, "detective" , 0 , "romance" , 0  ,"fantasy" , 0]
    scores_dic = Convert(scores)

    

    for i in range(testsetStart ,len(ids)):
        try:
            words = longy(attrib_words, get_book_words(ids[i]))
            predict = longest(words, attrs )

        except:
            print("rossz")
            predict = "non"
        if (predict == "non"):
            testset = testset - 1
            print("meret csokkentve")
        else:
            if predict == genre:
                print("helyes")
            else:
                print("helytelen")
            scores_dic[predict] = scores_dic[predict] + 1
    
    for i in scores_dic:
        scores_dic[i] = scores_dic[i] / testset

    print(testset)

    return scores_dic

def print_genre_attribes(attrs):
    print()
    print("scifi: ",attrs["scifi"])
    print()
    print("western: ",attrs["western"])
    print()
    print("detective: ",attrs["detective"])
    print()
    print("romance: ",attrs["romance"])
    print()
    print("fantasy: ",attrs["fantasy"])

def full_test(idf_for_book,attrib_words, genres, attrs):
    out = {}
    for genre in genres:
        out[genre] = test(genre, idf_for_book, attrib_words, attrs, numero)

    return out

#az attrnum-ban megadott mennyiségű legmagasabb tfidf értékűzz      
def attrs_correction(attrs, attrnum, genres):
    correct_attrs = {}

    for genre in genres:
        sorted_genre_attrs = sort_dict(attrs[genre])
        trimmed_genre_attrs = dict(list(sorted_genre_attrs.items())[:attrnum])
        correct_attrs[genre] = trimmed_genre_attrs

    return correct_attrs

def summarize(end_dic, genres):

    out = {}

    for genre in genres:
        out_line = {}
        for genre in genres:
            out_line[genre] = 0
        out[genre] = out_line
    

    for genre in genres:
        for num in end_dic:
            result = num[genre].value()
            res = {x: out[genre].get(x, 0) + result.get(x, 0)
                for x in set(out[genre]).union(result)}
            out[genre] = res
                
    print(out)
    
def create_genre_idf(genres, numero):
    
    all_genre_idf = {}

    for genre in genres:
        print(genre)
        genre_idf = {}
        fname_read = genre + "_" + str(numero) + "_testset.txt"
        text_file_full = open("/home/balint/Asztal/onlab/genres/bags/testsets/" + fname_read)
        text_full_ids = text_file_full.read()

        ids = word_tokenize(text_full_ids)

        setlen = len(ids)

        genre_book_words = {}

        for id in ids:
            id_words = get_book_words_only(id)
            for word in id_words:
                if word in genre_book_words.keys():
                    genre_book_words[word] = genre_book_words[word] + 1
                else:
                    genre_book_words[word] = 1

        for w in genre_book_words:
            genre_idf[w] = math.log10( float(setlen) / float(genre_book_words[w]))
        print(genre_idf)
        print()
        all_genre_idf[genre] = genre_idf


    return all_genre_idf
    print()

#print(create_tf_matrix())
#print(create_documents_per_words().keys())
#print(create_idf_matrix(create_documents_per_words(), 5))
#print(create_tf_idf_matrix(create_tf_matrix(), create_idf_matrix(create_documents_per_words(), 5)))



#print(numero_test)
#summarize(numero_test, genres)


attrs_per_genre = 30
genres = readGenres()

numero_test = {}

for numero in range(2):
    all_genre_idf = create_genre_idf(genres, numero)
    print(all_genre_idf["western"])

    
#print(test("fantasy", idf_for_book, merged_words, finale_attrs))
#full_test(idf_for_book, merged_words, genres,finale_attrs)
