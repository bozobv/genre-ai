import numpy as np
import pandas as pd
import glob

from collections import Counter
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

import seaborn as sns
sns.palplot(sns.color_palette())

green = sns.color_palette()[2]
red = sns.color_palette()[3]

df = pd.read_pickle("/home/balint/Asztal/onlab/gutenberg/metadata/bookshelves.p")

import sys
sys.path.append("/home/balint/Asztal/onlab/gutenberg/src")
from metaquery import meta_query
mq = meta_query(path="/home/balint/Asztal/onlab/gutenberg/metadata/metadata.csv", filter_exist=False)


# restrict to books we have
bookswehave = mq.df.id.values
new_idx = (np.intersect1d(df.index, bookswehave))
df = df.loc[new_idx]

print(df.shape)