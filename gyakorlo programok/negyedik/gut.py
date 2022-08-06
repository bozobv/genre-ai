import os
from urllib import request
import nltk
import re

title = 'Ulysses'
author = 'James Joyce'
url = 'https://www.gutenberg.org/files/4300/4300-0.txt'
path = 'corpora/canon_texts/'

# Check if the file is stored locally
filename = '/home/balint/Dokumentumok/corpora/canon_texts/' + title
if os.path.isfile(filename) and os.stat(filename).st_size != 0:
        print("{title} file already exists".format(title=title))
        with open(filename, 'r') as f:
            raw = f.read()

else:
    print("{title} file does not already exist. Grabbing from Project Gutenberg".format(title=title))
    response = request.urlopen(url)
    raw = response.read().decode('utf-8-sig')
    print("Saving {title} file".format(title=title))
    with open(filename, 'w') as outfile:
        outfile.write(raw)
