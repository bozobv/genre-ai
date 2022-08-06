import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


#text_file = open("test")
#text = text_file.read()

sia = SentimentIntensityAnalyzer()
out = sia.polarity_scores("Late and leaky. How the UK failed to impose an effective quarantine system ")


print(out)