import nltk
import glob
import csv 
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score

def sentiment(sentence,count): 
    ana = SentimentIntensityAnalyzer() 
    sentimental = ana.polarity_scores(sentence)
    print("{} Sentence is ".format(count))
    if sentimental['compound'] >= 0.05 :
        print("Positive")
        print()
        return 1
    else:
        print("Negative")
        print()
        return 0
ypred=[]
data=pd.read_excel("C://Users//navan//Desktop//DL//Data.xlsx")
x=data.iloc[:,0]
y=data.iloc[:,1]
count=1
for i,j in zip(x,y):
    p=sentiment(i, count)
    count=count+1
    ypred.append(p)
        
confusion_matrix_output =confusion_matrix(y, ypred)
print("Confusion Matrix: ")
print(confusion_matrix_output)
print()
print("Accuracy : ")
print(accuracy_score(y,ypred)*100)


    
