import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


sentence = "It is the largest tree I have ever seen in my entire hard life"

tokenized_words = word_tokenize(sentence)

for words in tokenized_words:
    tagged_words = nltk.pos_tag(tokenized_words)
    
grammar = "NP : {<DT>?<JJS>?<JJ>*<NN>} "
parser = nltk.RegexpParser(grammar)

output = parser.parse(tagged_words)
print (output)

output.draw()