import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import spacy
import os
import datetime
import multiprocessing as mp


def list2file(name, parsedlist):
    """Function to write list items to text file."""
    newfile = open(name, "w+")
    for parseditem in parsedlist:
        newfile.write(str(parseditem) + "\n")
    newfile.close()
    print("written to %s" % name)

def nltk_pp_toknpos(namencorp):
    """ Function used to map when parallelizing with nltk"""
    name, corp = namencorp
    return name, nltk.pos_tag(nltk.word_tokenize(corp))

def spacy_pp_toknpos(namencorp):
    """ Function used to map when parallelizing with spacy"""
    name, corp = namencorp
    doc = spnlp(corp, disable=['parser', 'ner'])
    postaglist = []
    for token in doc:
        postaglist.append(str((token.text, token.pos_)))
    return name, postaglist

# spacy initialization
spnlp = spacy.load("en_core_web_sm")
spnlp.max_length = 10000000

if __name__ == '__main__':

    # test strings and data to be used
    teststring1 = "All work and no play makes jack a dull boy, all work and no play"
    teststring2 = "Apple is looking at buying U.K. startup for $1 billion."

    # loading corpus (all text files) into string variable
    datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')

    strings = ''

    textfiles = [file for file in os.listdir(datapath) if file.endswith('.txt')]

    corpus = ''
    corpuslist = []

    for filename in textfiles:
        file = open(os.path.join(datapath, filename), "r")
        filestr = file.read()
        corpus = corpus + filestr
        corpuslist.append(filestr)



    ### Testing NLTK  with test strings ###
    print("NLTK test string results")

    now = datetime.datetime.now()
    tokenizedlist = word_tokenize(teststring1)
    proctime = datetime.datetime.now() - now
    print("teststring1 nltk tokens:")
    print(tokenizedlist)
    print("teststring1 nltk tokenizing time: {}".format(proctime))

    now = datetime.datetime.now()
    tokenizedlist = word_tokenize(teststring2)
    proctime = datetime.datetime.now() - now
    print("teststring2 nltk tokens:")
    print(tokenizedlist)
    print("teststring2 nltk tokenizing time: {}".format(proctime))

    tokenizedlist = word_tokenize(corpus)
    list2file('nltktokenizedwords.txt', tokenizedlist)

    ### Testing spacy with test strings ###


    print("spacy test string results")

    now = datetime.datetime.now()
    doc1 = spnlp(teststring1)
    proctime = datetime.datetime.now() - now
    tokenizedlist1 = [str(token) for token in doc1]
    print("teststring1 spacy tokens:")
    print(tokenizedlist1)
    print("teststring1 spacy tokenizing time: {}".format(proctime))

    now = datetime.datetime.now()
    doc2 = spnlp(teststring2)
    proctime = datetime.datetime.now() - now
    tokenizedlist2 = [str(token) for token in doc2]
    print("teststring2 spacy tokens:")
    print(tokenizedlist2)
    print("teststring2 spacy tokenizing time: {}".format(proctime))
    print("\n\n")

    ### Testing NLTK  with full corpus ###
    print("NLTK corpus results")

    ### Tokenizing ####
    now = datetime.datetime.now()
    tokenizedlist = word_tokenize(corpus)
    proctime = datetime.datetime.now() - now
    list2file('nltktokenizedwords.txt', tokenizedlist)
    print("NLTK corpus tokenizing time: {}".format(proctime))

    #### POS tagging ####
    now = datetime.datetime.now()
    postaglist = nltk.pos_tag(tokenizedlist)
    proctime = datetime.datetime.now() - now
    print("NLTK corpus POS tagging time: {}".format(proctime))
    list2file('nltkpostags.txt', postaglist)

    ### Stemming ####
    ps = PorterStemmer()
    now = datetime.datetime.now()
    stemlist = [ps.stem(word) for word in tokenizedlist]
    proctime = datetime.datetime.now() - now
    list2file('nltkwordstems.txt', stemlist)
    print("NLTK corpus stemming time: {}".format(proctime))


    ### Parallelizing ####
    corpdict = {textfiles[i]: corpuslist[i] for i in range(len(textfiles))}

    now = datetime.datetime.now()
    with mp.Pool() as pool:
        tokens = pool.map(nltk_pp_toknpos, corpdict.items())
    proctime = datetime.datetime.now() - now
    print("NLTK tokenization and POS tagging with parallel processing time: {}".format(proctime))
    list2file('nltkpptoknpos.txt', tokens)


    ### Testing spacy with full corpus ###
    print("spacy test string results")

    #### Tokenizing ####

    count = 0
    now = datetime.datetime.now()
    doclist = []
    disabled = spnlp.disable_pipes("parser", "tagger", "ner")
    for corp in corpuslist:
        doc = spnlp(corp)
        doclist.append(doc)
        count = count+1
        print("corpus {} completed.".format(count))
    proctime = datetime.datetime.now() - now
    spacytokenizetime = proctime
    print("spacy corpus tokenizing time: {}".format(proctime))

    tokenizedlist = []
    for doc in doclist:
        for token in doc:
            tokenizedlist.append(str(token))

    list2file("spacytokenized.txt", tokenizedlist)
    disabled.restore()


    #### POS tagging ####
    disabled = spnlp.disable_pipes("parser", "ner")
    now = datetime.datetime.now()
    doclist = []
    count=0
    for corp in corpuslist:
        doc = spnlp(corp)
        doclist.append(doc)
        count = count+1
        print("corpus {} POS tagging completed.".format(count))
    proctime = datetime.datetime.now() - now
    # cannot disable tokenizer so subtracting tokenized time from tokenized and tagging time
    print("spacy corpus POS tagging time: {}".format(proctime-spacytokenizetime))
    postaglist = []
    for doc in doclist:
        for token in doc:
            postaglist.append(str((token.text, token.pos_)))
    list2file('spacypostags.txt', postaglist)

    #### Parallelizing ####
    corpdict = {textfiles[i]: corpuslist[i] for i in range(len(textfiles))}

    now = datetime.datetime.now()
    with mp.Pool() as pool:
        tokens = pool.map(spacy_pp_toknpos, corpdict.items())
    proctime = datetime.datetime.now() - now
    print("spacy tokenization and POS tagging with parallel processing time: {}".format(proctime))
    list2file('spacypptoknpos.txt', tokens)