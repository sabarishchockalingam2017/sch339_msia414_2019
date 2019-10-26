import nltk
import os
import gensim
from gensim.models import Word2Vec
from numpy import dot
from numpy.linalg import norm

def cosine_distance (model, word,target_list) :
    """ Function to calculate cosine similarity between a word and words in a given list."""
    cosine_dict ={}
    word_list = []
    a = model[word]
    for item in target_list :
        if item != word :
            b = model [item]
            cos_sim = dot(a, b)/(norm(a)*norm(b))
            cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list

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
for corp in corpuslist:
    # tokenizing
    tokens = nltk.tokenize.word_tokenize(corp)

    # normalizing
    # filter out non-alphabetic
    words = [w for w in tokens if w.isalpha()]
    # converting to lower case
    words = [w.lower() for w in words]
    normalizedlist.append(" ".join(words))

# writing to text file
newfile = open("normalizedtokens.txt", "w+")

for normcorp in normalizedlist:
    newfile.write(normcorp)
    newfile.write("\n")

newfile.close()

# code to read file back in
readfile = open("normalizedtokens.txt", "r")

allcorpstring = readfile.read()
corplist = allcorpstring.split("\n")
normalizedlist = [nltk.tokenize.word_tokenize(corp) for corp in corplist]

# change model settings
model = Word2Vec(normalizedlist,
                 min_count=1,
                 size=50,
                 workers=3,
                 window=3,
                 sg=0)

savename = "word2vecsg0size50.model"
model.save(savename)

model = Word2Vec.load(savename)

testlist = ['secular', 'religion', 'society', 'atheism',
            'agnosticism', 'email', 'password', 'software',
            'computer', 'good', 'bad']

for word in testlist:
    print("\'{}\' cosine similarities: ".format(word))
    print(cosine_distance(model, word, testlist))
    print("\'{}\' euclidean similarities: ".format(word))
    print(model.most_similar(word))
