import numpy as np
import pandas as pd
import os,sys

from collections import Counter
import matplotlib.pyplot as plt

path_gutenberg = '/home/balint/Asztal/onlab/gutenberg'
src_dir = '/home/balint/Asztal/onlab/gutenberg-analysis/src'
sys.path.append(src_dir)

from data_io import get_book


#dataset = pd.read_csv('/home/balint/Asztal/onlab/gutenberg/metadata/metadata.csv', encoding="ISO-8859-1")
#pg_id = 'PG9932' 

#level = 'counts'
#dict_word_count = get_book(pg_id, level=level)
#print(dict_word_count)

dataset = pd.read_csv('/home/balint/Asztal/onlab/gutenberg/metadata/metadata.csv', encoding="ISO-8859-1")
#out = dataset.groupby('subjects')["title"].count()
#out_id = dataset.loc[dataset["id"] == pg_id, "subjects"]
#df = dataset[["id" , "subjects"]]
out2 = dataset[dataset['subjects'].str.contains('ndian')]
out = out2[["id" , "subjects"]]
#for col_name, data in df.items():
#    if (data.str.contains('Art') == True)
#        print("col _name:", col_name, "\ndata:", data)

#print(out.str.contains('Art', regex = False))
#print(out.to_string())
print(out)




