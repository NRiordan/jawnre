from flask import Flask, request
import random
import requests
import pickle
import make_model
import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from string import punctuation
from PyDictionary import PyDictionary


# Initialize your app and load your pickled models.
#================================================
# init flask app
app = Flask(__name__)

# load the pickled model from build_model.py
m = pickle.load(open('model.pkl', 'r'))

lookups = pickle.load(open('lookups.pkl', 'r'))
top5k = lookups[0]
top5k_dict = lookups[1]
rev_map_dict = lookups[2]

snowball = SnowballStemmer('english')



# Homepage with form on it.
#================================================
@app.route('/')
def index():
    return '''
    <form action="/predict" method='POST' >
        <input type="text" name="user_input" />
        <input type="submit" />
    </form>
    '''

# Once submit is hit, pass info into model, return results.
#================================================
@app.route('/predict', methods=['POST'])
def predcit():

    # get data from request form
    data = request.form['user_input']

    # convert data from unicode to string
    data = str(data)

    # make prediction based on new data
    pred = pred_genre(data)

    # return a string format of that prediction to the html page
    return pred


def transform_text(string, cutoff=100):
    # transform input text so that it matches top 5000 vector format
    string = string.replace("\xe2\x80\x99","'")
    string = string.replace("\xe2\x80\xa6","...")
    string = tokenize(string.strip(punctuation))

    # create empty array to hold counts of each of top 5k words
    to_fill = np.zeros(5001)

    for word in string:
        if word in top5k_dict:
            # only use word if its beyond cutoff (ie exclude top X most popular words)
            if top5k_dict[word] >= cutoff:
                to_fill[top5k_dict[word]] += 1

    # normalize counts using total number of words in input string
    to_fill /= len(string)*1.0
    return to_fill


def pred_genre(string):
    vector = transform_text(string)
    vector = vector.reshape(1, -1)
    prob_list = np.argsort(m.predict_proba(vector))

    # using predicted class order generates class/genre name as string
    top_genre = m.classes_[prob_list][0][::-1][0]

    # placeholder string for response to user
    answer = 'Sounds like some sweet sweet {} to me!'.format(top_genre)

    return answer


def tokenize(doc):
    # a simple tokenizing function
    # returns list of stemmed versions of words in doc
    return [snowball.stem(word) for word in word_tokenize(doc.lower())]


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
