import requests
from bs4 import BeautifulSoup
import nltk
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
from nltk import ngrams, FreqDist
from collections import Counter
le=WordNetLemmatizer()
#a
file=open("nlp_input.txt",encoding="utf8", errors='ignore')
fileToken=open("Token.txt","w+")
filele=open("Lemmatizer.txt","w+")
fileTri=open("TRI.text","w+")
trigramsOutput = []
n=0

for i in file.readlines():
    n=n+1
#b Tokenize the text into words and apply lemmatization
    splitWord = nltk.word_tokenize(i)
    splitSen=nltk.sent_tokenize(i)

    for m in ngrams(splitWord, 3):
        trigramsOutput.append(m)

    filele.write('[')
    for j in splitWord:
        l = le.lemmatize(j)
        filele.write(l)
        filele.write(',')
    fileToken.write(str(splitWord))
    filele.write(']')

file.close()
fileToken.close()
filele.close()
fileTri.write(str(trigramsOutput))
fileTri.close()
wordFreq = FreqDist(trigramsOutput)
# Getting Most Common Words and Printing them - Will get the Counts from top to least
top10 = wordFreq.most_common(10)
print("Top 10 triGrams : \n", top10)

file=open("nlp_input.txt",encoding="utf8", errors='ignore')
x=file.read()
splitSen=nltk.sent_tokenize(x)

concatenate=[]
for ((e,d,f),len) in top10:
    a=e+" "+d+" "+f
    for sentence in splitSen:
        if(sentence.find(a)>0):
            concatenate.append(sentence)
            print(sentence)
print(concatenate)
