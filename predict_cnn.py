from keras.preprocessing.sequence import pad_sequences
import gensim
from keras.models import load_model
import numpy as np

# input cnn model
modelpath = './models/yelpcnn6.model'
inputdata = './inputtext.txt'
outfilepath = './predictions/cnnbestout.json'
dictpath = './yelp.dict'
MAX_SEQUENCE_LENGTH = 197



def token_to_index(token, dictionary):
    """
    Given a token and a gensim dictionary, return the token index
    if in the dictionary, None otherwise.
    Reserve index 0 for padding.
    """
    if token not in dictionary.token2id:
        return None
    return dictionary.token2id[token] + 1

def texts_to_indices(text, dictionary):
    """
    Given a list of tokens (text) and a gensim dictionary, return a list
    of token ids.
    """
    result = list(map(lambda x: token_to_index(x, dictionary), text))
    return list(filter(None, result))


if __name__ == '__main__':
    # loading files and transforming input text
    model = load_model(modelpath)

    # reading in text from file
    txtfile = open(inputdata, mode='r', encoding='utf8')

    txtlist = txtfile.readlines()
    txtfile.close()

    # loading dictionary
    dictionary = gensim.corpora.Dictionary.load(dictpath)
    # normalizing and tokenizing
    tokens = [t.strip(',.&').lower().split() for t in txtlist]
    tokens = list(map(lambda y: list(filter(lambda x: x.isalpha(), y)), tokens))

    # converting text to dictionary indices
    texts_indices = list(map(lambda x: texts_to_indices(x, dictionary), tokens))
    # pad or truncate the texts
    x_data = pad_sequences(texts_indices, maxlen=int(MAX_SEQUENCE_LENGTH))
    conf = model.predict(x_data)
    # getting labels that have maximum probability/confidence for each input text
    predictions = [list(probs).index(np.max(probs)) for probs in conf]

    outfile = open(outfilepath,'w')
    for p, c in zip(predictions, conf):
        writestr = '{' + "'label':{}, 'confidence':{}".format(p, c)+'}'
        outfile.write(writestr)
        outfile.write("\n")
    outfile.close()
    print('Predicitions outputted to: {}'.format(outfilepath))