import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt 

text_file = open('/home/balint/Asztal/onlab/gyakorlo programok/masodik/lilprince')
#text = "This is a very good day, my dear!"
text = text_file.read()
words = word_tokenize(text)


words_no_punc = []
clean_words = []
stopwords = stopwords.words("english")

print(words)

for w in words:
    if w.isalpha():
        words_no_punc.append(w.lower())
    
for w in words_no_punc:
    if w not in stopwords:
        clean_words.append(w)

#print(words_no_punc)
#print("\n")

#print (clean_words)
print (clean_words)

fdist = FreqDist(clean_words)
fdist.plot(10)
