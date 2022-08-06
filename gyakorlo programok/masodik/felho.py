import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt 

text_file = open("/home/balint/Asztal/onlab/gyakorlo programok/harmadik/lilprince")
text = text_file.read()

wordcloud = WordCloud().generate(text)
wordcloud.background_color = "white"
max_font_size = 4

plt.figure(figsize = (12, 12)) 
plt.imshow(wordcloud) 


plt.axis("off") 
plt.show()