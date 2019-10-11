import nltk
import spacy
import os
import re

# loading corpus (all text files) into string variable
datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')

strings=''

textfiles = [file for file in os.listdir(datapath) if file.endswith('.txt')]

corpus = ''

for filename in textfiles[0:2]:
    file = open(os.path.join(datapath, filename), "r")
    corpus = corpus + file.read()


def parsed2file(name,parsedlist):
    newfile = open(name, "w+")
    for parseditem in parsedlist:
        newfile.write(parseditem + "\n")
    newfile.close()
    print("written to %s"%name)

# finding emails
emailpatt = re.compile(r"[a-zA-Z0-9._-]*[@]\w*[.].[a-zA-Z0-9._-]*")

emailsparsed = emailpatt.findall(corpus)
print(emailsparsed)
parsed2file("corpusemails.txt", emailsparsed)

# finding dates
# test cases
teststrings = ["14 Apr 1993",
               "April 20th 2019",
               "20th April 2019",
               "11-01-1993",
               "11\\01\\1993",
               "11/01/1993"]

teststringconcat = " ".join(teststrings)

# different times of date patterns assumed
datepatt1 = re.compile(r"(?<=\s)[0-3]?[1-9][-/\\][0-3]?[1-9][-/\\]\d{4}")
datepatt2 = re.compile(r"(?<=Date: ).*\d")
datepatt3 = re.compile(r"(?:\d+)\s(?:Jan(?:uary)"
                       r"?|Feb(?:ruary)"
                       r"?|Mar(?:ch)"
                       r"?|Apr(?:il)"
                       r"?|May"
                       r"?|Jun(?:e)"
                       r"?|Jul(?:y)"
                       r"?|Aug(?:ust)"
                       r"?|Sep(?:tember)"
                       r"?|Oct(?:ober)"
                       r"?|Nov(?:ember)"
                       r"?|Dec(?:ember))\s(?:\d*)")

datepatt4 = re.compile(r"(?:\d+)(?:st?|nd|rd|th)"
                       r"\s(?:Jan(?:uary)"
                       r"?|Feb(?:ruary)"
                       r"?|Mar(?:ch)"
                       r"?|Apr(?:il)"
                       r"?|May"
                       r"?|Jun(?:e)"
                       r"?|Jul(?:y)"
                       r"?|Aug(?:ust)"
                       r"?|Sep(?:tember)"
                       r"?|Oct(?:ober)"
                       r"?|Nov(?:ember)"
                       r"?|Dec(?:ember))\s(?:\d*)")

pattlist = [datepatt1, datepatt2, datepatt3, datepatt4]

def parsealldates(inpstring):
    datesfound = []
    for pattern in pattlist:
        pattfound = pattern.findall(inpstring)
        for date in pattfound:
            datesfound.append(date)
    return datesfound

print("Parsed from tests: ")
parseddates = parsealldates(teststringconcat)
print(parseddates)

parsed2file("testdates.txt", parseddates)

print("Parsed from corpus: ")
parseddates = parsealldates(corpus)
print(parseddates)

parsed2file("corpusdates.txt", parseddates)

