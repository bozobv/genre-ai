import nltk
from nltk import sent_tokenize
from nltk import word_tokenize

text_file = open("test")

text = text_file.read()

sentences = sent_tokenize(text)

print(len(sentences))

#print(sentences)