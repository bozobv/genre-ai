import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import twitter_samples 
 
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
 
analyzer = SentimentIntensityAnalyzer()

#print(all_negative_tweets[20])
#print(analyzer.polarity_scores(all_negative_tweets[20]))


my_labels = [1]*len(all_positive_tweets)
negative_labels = [0]*len(all_negative_tweets)
my_labels.extend(negative_labels)
 
all_positive_tweets.extend(all_negative_tweets)
 
df = pd.DataFrame({'tweets' : all_positive_tweets,'my_labels' : my_labels})
df['neg'] = df['tweets'].apply(lambda x:analyzer.polarity_scores(x)['neg'])
df['neu'] = df['tweets'].apply(lambda x:analyzer.polarity_scores(x)['neu'])
#df['pos'] = df['tweets'].apply(lambda x:analyzer.polarity_scores(x)['pos'])
#df['compound'] = df['tweets'].apply(lambda x:analyzer.polarity_scores(x)['compound'])
                                     

 
print(df) 