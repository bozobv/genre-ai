import numpy as np
import pandas as pd
import os,sys

from collections import Counter
import matplotlib.pyplot as plt

path_gutenberg = '/home/balint/Asztal/onlab/gutenberg'
src_dir = '/home/balint/Asztal/onlab/gutenberg-analysis/src'
sys.path.append(src_dir)



def sort_by_subject(
    search_tags,
    path_to_raw_file,
    path_to_clean_file
):

    dataset = pd.read_csv('/home/balint/Asztal/onlab/gutenberg/metadata/metadata.csv', encoding="ISO-8859-1")

    out2 = dataset[dataset['subjects'].str.contains(search_tags)]
    out2 = out2[out2['language'].str.contains("en")]
    out = out2[["id"]]

    out.to_csv(path_to_raw_file)

    infile = path_to_raw_file
    outfile = path_to_clean_file

    with open(infile) as fin, open(outfile, "w+") as fout:
        for line in fin:
            splitting = line.split(',')
            print(splitting)
            fout.write(splitting[1])


sort_by_subject( 'detective|Detective|mistery|Mistery' , "/home/balint/Asztal/onlab/genres/raw/detective.csv","/home/balint/Asztal/onlab/genres/clean/detective.txt" ,)

sort_by_subject( 'Science fiction' , "/home/balint/Asztal/onlab/genres/raw/scifi.csv","/home/balint/Asztal/onlab/genres/clean/scifi.txt" ,)

sort_by_subject( 'Fantasy|fantasy|Fairy tales' , "/home/balint/Asztal/onlab/genres/raw/fantasy.csv","/home/balint/Asztal/onlab/genres/clean/fantasy.txt" ,)

sort_by_subject( 'Romance|Love stories' , "/home/balint/Asztal/onlab/genres/raw/romance.csv","/home/balint/Asztal/onlab/genres/clean/romance.txt" ,)

sort_by_subject( 'Cowboy|cowboy|Western stories|Indians of North America -- Fiction' , "/home/balint/Asztal/onlab/genres/raw/western.csv","/home/balint/Asztal/onlab/genres/clean/western.txt" ,)



