import numpy as np
import pandas as pd
import os,sys

from collections import Counter
import matplotlib.pyplot as plt

infile = "/home/balint/Asztal/onlab/fantasy_id.csv"
outfile = "/home/balint/Asztal/onlab/fantasy.txt"

delete_list = [",True"]
with open(infile) as fin, open(outfile, "w+") as fout:
    for line in fin:
        for word in delete_list:
            line = line.replace(word, "")
        fout.write(line)