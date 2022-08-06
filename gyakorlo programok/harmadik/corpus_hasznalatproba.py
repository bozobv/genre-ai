import nltk
from nltk.corpus import brown
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

#print(brown.categories())
text_file = nltk.corpus.gutenberg.words('austen-persuasion.txt')

print(text_file[0:10])

files = nltk.corpus.gutenberg.fileids()
print(files)