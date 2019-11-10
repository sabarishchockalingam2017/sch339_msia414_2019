import pickle
# input svm model
modelpath = './models/svmbest.model'
# input associated tfdif vectorizer to transform text
vectpath = './models/svmbestvect.vect'
inputdata = './inputtext.txt'
outfilepath = './predictions/svmbestout1.json'

if __name__ == '__main__':
    # loading files and transforming input text
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
    print('Predicitions outputted to: {}'.format(outfilepath))