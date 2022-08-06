import numpy as np
import pandas as pd
import os, sys

from collections import Counter
import matplotlib.pyplot as plt

path_gutenberg = os.path.join(os.pardir,os.pardir,'gutenberg')
src_dir = '/home/balint/Asztal/onlab/gutenberg-analysis/src'
sys.path.append(src_dir)
src_dir = os.path.join(os.pardir,'src')
sys.path.append(src_dir)
from data_io import get_book

pg_id = 'PG2701' ## moby dick


level = 'counts'
dict_word_count = get_book(pg_id, level=level)
print(dict_word_count)

