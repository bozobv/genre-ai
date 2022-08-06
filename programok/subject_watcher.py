import numpy as np
import pandas as pd
import os,sys

from collections import Counter
import matplotlib.pyplot as plt

#import sys
sys.path.insert(0, "/home/balint/Asztal/onlab/gutenberg-analysis/src")


path_gutenberg = '/home/balint/Asztal/onlab/gutenberg'

src_dir = '/home/balint/Asztal/onlab/gutenberg-analysis/src'
sys.path.append(src_dir)
from data_io import get_book

sys.path.append(os.path.join(path_gutenberg,'src'))
from metaquery import meta_query

#mq = meta_query(path=os.path.join(path_gutenberg,'metadata','metadata.csv'))
#mq.filter_subject("Spy stories",how='any')
#mq.filter_lang('en', how = 'only')

dataset = pd.read_csv('/home/balint/Asztal/onlab/gutenberg/metadata/metadata.csv', encoding="ISO-8859-1")
#out = dataset.groupby('subjects')["title"].count()
out = dataset[["id", "subjects", "language"]]



#fájlba mentés
#np.savetxt("/home/balint/Asztal/onlab/subject/subjects", out, fmt = '%s')

print(out)
#print(mq.df[id])
