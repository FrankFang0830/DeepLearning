import nltk
nltk.download("wordnet")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
# print(wn.synsets('phone'))
# for i in wn.synsets('phone'):
#     print(str(i.definition))


# sentence="I am so busy until next weekend. Pls be next week if you want to hang out with me"
# stokens=nltk.sent_tokenize(sentence)
# tokens=nltk.word_tokenize(sentence)
# tagging=nltk.pos_tag(tokens)
# for i in stokens:
#     print(i)
# for j in tokens:
#     print(j)
# print(tagging)


# pStemmer=PorterStemmer()
# print(pStemmer.stem('better'))
nltk.download('words')
nltk.download('maxent_ne_chunker')
from nltk import wordpunct_tokenize,pos_tag,ne_chunk
sentence="John will back to Beijing and work at Google"
print(ne_chunk(pos_tag(wordpunct_tokenize(sentence))))