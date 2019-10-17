import nltk
import os
from nltk.stem import WordNetLemmatizer
import gensim

# loading corpus (all text files) into string variable
datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

textfiles = [file for file in os.listdir(datapath) if file.endswith('.txt')]

corpus = ''
corpuslist = []

for filename in textfiles:
    file = open(os.path.join(datapath, filename), "r")
    filestr = file.read()
    corpus = corpus + filestr
    corpuslist.append(filestr)

normalizedlist = []
for corp in corpuslist[0:2]:
    # tokenizing
    tokens = nltk.tokenize.word_tokenize(corp)

    # normalizing
    # filter out non-alphabetic
    words = [w for w in tokens if w.isalpha()]
    # converting to lower case
    words = [w.lower() for w in words]
    normalizedlist.append(" ".join(words))
    print(" ".join(words))
