import os
import json
import re
import numpy as np
import pandas as pd
import fasttext
from sklearn.model_selection import KFold


def ft_accuracy(datanlabels, ftmodel):
    """ Function to calculate accuracy for fast text resutls.
    Arguments:
        datanlabels: list of strings startign with '__label__xxx' where xxx is
        the rating, and review text is followed.
        ftmodel: model outputted by fasttext """

    correct = 0
    for datapt in datanlabels:
        pred = ftmodel.predict(re.match(r'__label__...\s(.*)', datapt).group(1))[0][0]
        actual = datapt.split(' ')[0]
        if pred == actual:
            correct = correct + 1
    accuracy = correct / len(datanlabels)
    return accuracy

# loading corpus (all text files) into string variable
datapath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

# getting names of json files
jsonfiles = [file for file in os.listdir(datapath) if file.endswith('.json')]
jsonpatt = re.compile(r'^(.*?).json')

# creating review df with relevant columns
revfile = open(os.path.join(datapath,'review.json'), mode='r', encoding='utf8')
readlim = 500000
revlist = [json.loads(next(revfile)) for line in range(readlim)]
revfile.close()


# creating a path to place formatted files matching fasttext requirements
preprocpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'preprocessed')
if not os.path.exists(preprocpath):
    os.makedirs(preprocpath)

# setting up to do cross validation
kf = KFold(n_splits=5, shuffle=True)

precisions = []
recalls = []
accuracies = []

# modelling with cross validation
for trainindex, testindex in kf.split(revlist):

    # formatting training set
    ftrevtrain = ['__label__'+str(revlist[ind]['stars'])+' '+revlist[ind]['text'].replace('\n', ' ')
                  for ind in trainindex]

    # writing formatted data to file
    ftformat = open(os.path.join(preprocpath, 'ftrevtrain.txt'), mode='w+', encoding='utf8')
    for rev in ftrevtrain:
        ftformat.write(rev)
        ftformat.write('\n')
    ftformat.close()

    # formatting test set
    ftrevtest = ['__label__' + str(revlist[ind]['stars']) + ' ' + revlist[ind]['text'].replace('\n', ' ')
                 for ind in testindex]

    # writing formatted test data to file
    ftformat = open(os.path.join(preprocpath, 'ftrevtest.txt'), mode='w+', encoding='utf8')
    for rev in ftrevtest:
        ftformat.write(rev)
        ftformat.write('\n')
    ftformat.close()

    # training fasttext model
    model = fasttext.train_supervised(input=os.path.join(preprocpath, 'ftrevtrain.txt'), lr=1.0)

    # predicting test set and evaluating performance
    testresult = model.test(os.path.join(preprocpath, 'ftrevtest.txt'), k=1)
    precisions.append(testresult[1])
    recalls.append(testresult[2])
    accuracies.append(ft_accuracy(ftrevtest, model))

print("Mean Precision: ", np.mean(precisions), "Precisions: ", precisions)
print("Mean Recall: ", np.mean(recalls), "Recalls: ", recalls)
print("Mean Accuracy: ", np.mean(accuracies), "Accuracies: ", accuracies)

