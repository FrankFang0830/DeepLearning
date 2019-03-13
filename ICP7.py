import requests
from bs4 import BeautifulSoup
import nltk
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
from nltk import wordpunct_tokenize,pos_tag,ne_chunk
from nltk.util import ngrams
from nltk import tag
# nltk.download("wordnet")
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


url = "https://en.wikipedia.org/wiki/Google"
source = requests.get(url)
html_text = source.text
soup = BeautifulSoup(html_text, "html.parser")
text=soup.get_text()
file=open('input.txt','w',encoding='utf-8')
file.write(text)
file.close()

file=open("input.txt","r")
fileToken=open("Token.txt","w+")
filePOST=open("POS.text",'w+')
fileStem=open("Stemming.text",'w+')
fileL=open("Lemmatizer.text",'w+')
fileNER=open("NER.text",'w+')
fileTRI=open("TRI.text",'w+')
pStemmer=PorterStemmer()
le=WordNetLemmatizer()
for line in file :
   sentence=file.read()
   sentence1=nltk.sent_tokenize(sentence)

   stokens=nltk.word_tokenize(sentence)
   ner=ne_chunk(pos_tag(wordpunct_tokenize(sentence)))
   tri = list(ngrams(stokens, 3))

   # stokens=['cats','showed']
   #stem = pStemmer.stem(str(stokens))
   #we can't make a string as a argument to stemming, we have to seprate it to get correct result
   #otherwise it can not recognize word.
   #Stemming and Lemmatization both generate the root form of the inflected words.
   #Stemming are more faster than lemmatization,but sometime it return to a fake root
   tag=nltk.pos_tag(stokens)
   for j in stokens:
      l=le.lemmatize(j)
      stem=pStemmer.stem(j)
      fileL.write(",")
      fileL.write(str(l))
      fileStem.write(",")
      fileStem.write(str(stem))
   fileToken.write(str(stokens))
   fileTRI.write(str(tri))
   filePOST.write(str(tag))
   fileNER.write(str(ner))


file.close()
fileToken.close()
filePOST.close()
fileStem.close()
fileNER.close()
fileL.close()