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

pg_id = 'PG2901' ## moby dick

sys.path.append(os.path.join(path_gutenberg,'src'))
from metaquery import meta_query


mq = meta_query(path = os.path.join(path_gutenberg,'metadata','metadata.csv'))

df = pd.read_pickle(os.path.join(path_gutenberg,'metadata','bookshelves.p'))

df_out = df.loc[df["Education_(Bookshelf)"] == True]
df_out = df_out[["Education_(Bookshelf)"]]

df_out.to_csv('/home/balint/Asztal/onlab/edu_id.csv')

print(df_out)
