import os
import json
import re
import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt

# loading corpus (all text files) into string variable
datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

# getting names of json files
jsonfiles = [file for file in os.listdir(datapath) if file.endswith('.json')]
jsonpatt = re.compile(r'^(.*?).json')

# loading json files into a dictionary
docdict = {}
for doc in jsonfiles:
    print("loading {}".format(doc))
    templist = []
    for line in open(os.path.join(datapath, doc), mode='r', encoding='utf8'):
        templist.append(json.loads(line))
    docname = jsonpatt.findall(doc)[0]
    docdict[docname] = templist
    print("appended {} list to dict".format(docname))

# creating dataframes to get summary statistics
dfdict = {}

for doc in jsonfiles:
    docname = jsonpatt.findall(doc)[0]
    tempdf = pd.DataFrame(data = docdict[docname])
    dfdict[docname] = tempdf
    print('created {} df'.format(docname))

# getting some preliminary stats for reviews data
print('Basic stats for reviews data: ')
print(dfdict['review'].describe(include='all'))

# taking only first 500k since entire dataset is too large (6mil+)
allreviewsstring = ' '.join(dfdict['review']['text'].head(500000))
# tokenizing
tokens = nltk.tokenize.word_tokenize(allreviewsstring)
# normalizing
# filter out non-alphabetic
words = [w for w in tokens if w.isalpha()]

avg_wordlen = np.mean([len(word) for word in words])

print("The average word length in yelp reviews is {} letters.".format(avg_wordlen))

# plotting label/rating distribution
revdf = dfdict['review'].head(500000)
revdf = revdf[['text','stars']]
revdf['stars'] = revdf['stars'].astype('category')
fig = plt.figure(figsize=(8,6))
revdf.groupby('stars').text.count().plot.bar(ylim=0)
plt.ylabel('Frequency')
plt.title('Label Distribution')
plt.savefig('rating_dist.jpg')
