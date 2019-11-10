import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import pickle

# loading corpus (all text files) into string variable
datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

# creating review df with relevant columns
revfile = open(os.path.join(datapath, 'review.json'), mode='r', encoding='utf8')
readlim = 500000
revlist = [json.loads(next(revfile)) for line in range(readlim)]
revdf = pd.DataFrame(data=revlist)
revdf = revdf[['text', 'stars']]
revdf['stars'] = revdf['stars'].astype('category')


# tfdif parameters
mindf = 5000
maxdf = 1.0 # set to 1.0 for no limit or 100% of documents
ngramrange = (1, 2)

# vectorizing corpus
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=mindf, norm='l2',
                        encoding='utf8', ngram_range=ngramrange,
                        stop_words='english',
                        max_df=maxdf)

features = tfidf.fit_transform(revdf.text).toarray()
labels = revdf.stars
print("features shape is: ")
print(features.shape)

# finding two most correlated words for first 10 reviews
N = 2
for review, rating in sorted(revdf.stars.items())[:10]:
    features_chi2 = chi2(features, labels == rating)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    print("Review: {} Rating: {}:".format(review+1, rating))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))

# fitting and scoring model (SVM)
CV = 5
model = LinearSVC()
accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
print("accuracy for this model was: {}".format(accuracies.mean()))

pickle.dump(model,open('./models/svmbest.model','wb'))
