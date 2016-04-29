from flask import Flask, url_for, request, render_template
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
key_words_dict = lookups[3]

snowball = SnowballStemmer('english')



# Homepage with form on it.
#================================================
@app.route('/', methods=['get', 'POST'])
@app.route('/index', methods=['get','POST'])
def index():
    pred = "created by neal riordan"
    suggestion = "for galvanize"
    last = " "
    return render_template('index.html', pred=pred, suggestion=suggestion, last=last)

# Once submit is hit, pass info into model, return results.
#================================================
@app.route('/predict', methods=['get','POST'])
def predict():

    # get data from request form
    data = request.form['user_input']

    # convert data from unicode to string
    data = str(data)

    if len(data) == 0:
        pred = "you didn't enter anything, brush yourself off and try again"
    else:
        # make prediction based on new data
        top_genre = pred_genre(data)
        pred = 'Sounds like some sweet sweet {} to me!'.format(top_genre)

        new_genre, replacement = make_suggestion(top_genre, data)
        suggestion = 'Want to make it a little more {}? Try adding "{}"'.format(new_genre, replacement)

    # return a string format of that prediction to the html page
    return render_template('index.html', pred=pred, suggestion=suggestion, last=data)


def make_suggestion(genre, string):
    string = tokenize(string.strip(punctuation))
    all_genres = list(m.classes_)

    all_genres.remove(genre)
    new_genre = np.random.choice(all_genres)

    word_options = key_words_dict[new_genre]
    replacement = np.random.choice(word_options)

    return new_genre, replacement

def transform_text(string, cutoff=100):
    # transform input text so that it matches top 5000 vector format
    string = string.replace("\xe2\x80\x99","'")
    string = string.replace("\xe2\x80\xa6","...")
    string = string.replace("\u2018","")
    string = string.replace("\u2019","")

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

    return top_genre


def tokenize(doc):
    # a simple tokenizing function
    # returns list of stemmed versions of words in doc
    return [snowball.stem(word) for word in word_tokenize(doc.lower())]


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
