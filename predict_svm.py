import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import pickle
import argparse

modelpath = './models/svmbest.model'
vectpath = './models/svmbestvect.vect'
inputdata = './inputtext.txt'
outfilepath = './predictions/svmbestout.json'

if __name__ == '__main__':
    model = pickle.load(open(modelpath, 'rb'))
    tfidf = pickle.load(open(vectpath, 'rb'))
    transd = tfidf.transform(open(inputdata, 'r'))
    predictions = model.predict(transd)
    conf = model._predict_proba_lr(transd)

    outfile = open(outfilepath,'w')
    for p, c in zip(predictions, conf):
        writestr = '{' + "'label':{}, 'confidence':{}".format(p, c)+'}'
        outfile.write(writestr)
        outfile.write("\n")
    outfile.close()
