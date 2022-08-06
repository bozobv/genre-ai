import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.data
import matplotlib.pyplot as plt 
from nltk.probability import FreqDist

def chunks(inlist, size):
    for i in range(0, len(inlist), size):
        yield inlist[i:i + size]

def avgchunk(inlist, chunksize):
    chunked = []
    chunked = chunks(inlist, chunksize)
    for chunk in chunked:
        yield sum(chunk) / len(chunk)
    

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
analyzer = SentimentIntensityAnalyzer()

text_file = open("/home/balint/Asztal/onlab/gyakorlo programok/masodik/lilprince")
text = text_file.read()



sentences = nltk.tokenize.sent_tokenize(text)
sia = SentimentIntensityAnalyzer()

#for s in sentences:
#    out.append(sia.polarity_scores(s))

print(len(sentences))
#my_labels = [0]*len(sentences)
df = pd.DataFrame({'sentence' : sentences})
df['neg'] = df['sentence'].apply(lambda x:analyzer.polarity_scores(x)['neg'])
#df['neu'] = df['sentence'].apply(lambda x:analyzer.polarity_scores(x)['neu'])
df['pos'] = df['sentence'].apply(lambda x:analyzer.polarity_scores(x)['pos'])
df['compound'] = df['sentence'].apply(lambda x:analyzer.polarity_scores(x)['compound'])

# cpm = []
# cpm = chunks(df['compound'], 200)

# avcpm = []
# for n in cpm:
#     avcpm.append(sum(n) / len(n))

avcpm = list(avgchunk(df['compound'], 200))
avneg = list(avgchunk(df['neg'], 200))
avpos = list(avgchunk(df['pos'], 200))

print(len(avcpm))


plt.plot(avcpm)
plt.plot(avneg)
plt.plot(avpos)

plt.show()


