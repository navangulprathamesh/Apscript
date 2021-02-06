import nltk
import glob
import csv 
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import pandas as pd
from sklearn.metrics import confusion_matrix

def sentiment(sentence,count): 
    ana = SentimentIntensityAnalyzer() 
    sentimental = ana.polarity_scores(sentence)
    print("{} Sentence is ".format(count))
    if sentimental['compound'] >= 0.05 :
        print("Positive")
        return 1
    elif sentimental['compound'] <= - 0.05 :
        print("Negative")
        return 0
    else :
        print("Neutral")
        return 2
ypred=[]
data=pd.read_excel("C://Users//navan//Desktop//DL//Data_train.xlsx")
x=data.iloc[:,0]
y=data.iloc[:,1]
count=1
for i,j in zip(x,y):
    p=sentiment(i, count)
    count=count+1
    ypred.append(p)
        
confusion_matrix_output =confusion_matrix(y, ypred)
print(confusion_matrix)


    
