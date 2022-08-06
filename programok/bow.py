import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt 
import pandas as pd
import operator

def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct

def whichGenre(pg_id):
    fname_read = '%s_text.txt'%pg_id
    #print("fname_read:  " + fname_read )
    try:
        filename = open("/home/balint/Asztal/onlab/gutenberg/data/text/" + fname_read, "r") 
        text = filename.read()

        words = word_tokenize(text)

        words_no_punc = []
        clean_words = []
        stw = stopwords.words("english")

        for w in words:
            if w.isalpha():
                words_no_punc.append(w.lower())
            
        for w in words_no_punc:
            if w not in stw:
                clean_words.append(w)


        df = pd.read_csv('/home/balint/Asztal/onlab/genres/bags/THE_BAG.csv')

     

        west = 0
        fant = 0
        scif = 0
        dete = 0
        roma = 0

        fdist = FreqDist(clean_words)

        #rangeNumber = len(fdist) // 30
        rangeNumber = 200

        for i in range(rangeNumber):
            df_cell = df.loc[df['word'] == fdist.most_common(rangeNumber)[i][0]]
            west = west+ df_cell['western'].sum() * fdist.most_common(rangeNumber)[i][1]
            fant = fant + df_cell['fantasy'].sum() * fdist.most_common(rangeNumber)[i][1]
            scif = scif + df_cell['scifi'].sum()* fdist.most_common(rangeNumber)[i][1]
            dete = dete + df_cell['detective'].sum()* fdist.most_common(rangeNumber)[i][1]
            roma = roma + df_cell['romance'].sum()* fdist.most_common(rangeNumber)[i][1]

        scores = ["western" , west  , "scifi" , scif, "detective" , dete , "romance" , roma  ,"fantasy" , fant]
        scores_dic = Convert(scores)
        return max(scores_dic.items(), key = operator.itemgetter(1))[0]
    except:
        print(fname_read + " hiba")
        return "non"


def genre_test_bow(genre):

    text_file_genre = open("/home/balint/Asztal/onlab/genres/clean/" + genre +".txt")

    text_genre_ids = text_file_genre.read()
    words = word_tokenize(text_genre_ids)

    testsetStart = len(words) * 4 // 5
    testset = len(words) - testsetStart
    print(testset)
    
    print("genre test: " + genre)    
    
    scores = ["western" , 0  , "scifi" , 0, "detective" , 0 , "romance" , 0  ,"fantasy" , 0]
    scores_dic = Convert(scores)


    for i in range(testsetStart ,len(words)):
        predict = whichGenre(words[i])
        if (predict == "non"):
            testset = testset - 1
            print("meret csokkentve")
        else:
            if predict != genre:
                print("wrong predict:" + predict)
            else:
                print("good predict")
            scores_dic[predict] = scores_dic[predict] + 1   
    
    for i in scores_dic:
        scores_dic[i] = scores_dic[i] / testset

    return scores_dic


lst= [genre_test_bow("western"),genre_test_bow("scifi"),genre_test_bow("detective"),genre_test_bow("romance"),genre_test_bow("fantasy")]

print("siker aránya western: ",  lst[0])
print("siker aránya scifi: ",  lst[1])
print("siker aránya detective: ",  lst[2])
print("siker aránya romance: ",  lst[3])
print("siker aránya fantasy: ",  lst[4])