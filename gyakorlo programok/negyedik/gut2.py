import os
from urllib import request
import nltk
import re

url = 'https://www.gutenberg.org/files/100/100-0.txt'
path = 'corpora/canon_texts/'

response = request.urlopen(url)
raw = response.read().decode('utf-8-sig')


print("Saving {title} file".format(title=title))
with open(filename, 'w') as outfile:
    outfile.write(raw)
