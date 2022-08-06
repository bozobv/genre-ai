import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt 


text_file = open("bible.txt")
text = text_file.read()

words = word_tokenize(text)


words_no_punc = []
clean_words = []
stopwords = stopwords.words("english")

for w in words:
    if w.isalpha():
        words_no_punc.append(w.lower())
    
for w in words_no_punc:
    if w not in stopwords:
        clean_words.append(w)

lemmatizer = WordNetLemmatizer()

lemmatized_words = []

for w in clean_words:
   lemmatized_words.append(lemmatizer.lemmatize(w))


fdist = FreqDist(lemmatized_words)
fdist.plot(20)

