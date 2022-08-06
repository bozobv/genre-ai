import numpy as np
import pandas as pd
import os,sys
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize

def weight(flname, lookup):
    filename = os.path.join("/home/balint/Asztal/onlab/genres/bags/" + flname + ".txt")

    idx = -1

    with open(filename) as myFile:
        for num, line in enumerate(myFile, 1):
            if lookup in line:
                idx = num
                break

    if idx < 0:
        return 0

    return (1001 - idx) / 1000

def createFrame(lookup):
    genres = ["western","fantasy","scifi","detective","romance"]

    return [lookup, weight(genres[0] ,lookup), weight(genres[1] ,lookup), weight(genres[2] ,lookup), weight(genres[3] ,lookup),weight(genres[4] ,lookup)]


filename = os.path.join("/home/balint/Asztal/onlab/genres/bags/all.txt")
text_file = open(filename) 
text = text_file.read()

toked_words = word_tokenize(text)

rows = []

for i in range(len(toked_words)):
    rows.append(createFrame(toked_words[i]))

df = pd.DataFrame(rows, columns=["word", "western","fantasy","scifi","detective","romance"])

df.to_csv("/home/balint/Asztal/onlab/genres/bags/THE_BAG.csv", index = False)