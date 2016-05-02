from flask import Flask, url_for, request, render_template
import random
import requests
import pickle
import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from string import punctuation


# Initialize flask app
app = Flask(__name__)

# Load the pickled model produced by make_model.py
m = pickle.load(open('model.pkl', 'r'))

# Load the various pickled lookup lists/dicts produced by make_model.py
lookups = pickle.load(open('lookups.pkl', 'r'))
top5k = lookups[0]              # List of top 5000 words in order
top5k_dict = lookups[1]         # key = word, value = index in top5k list
rev_map_dict = lookups[2]       # key = stemmed word, value = unstemmed version
key_words_dict = lookups[3]     # key = genre, value = list of 'defining' words

# Instantiate the stemmer
snowball = SnowballStemmer('english')



# Homepage with form on it.
@app.route('/', methods=['get', 'POST'])
@app.route('/index', methods=['get','POST'])
def index():
    # Text to be replaced by model prediction
    pred = "created by neal riordan"

    # Text to be replaced by suggestion to alter genre
    suggest = "for galvanize"

    # Text to be replaced by last string entered (for user reference)
    last = " "

    return render_template('index.html', pred=pred, suggest=suggest, last=last)

# Once submit is hit, pass info into model, return results.
@app.route('/predict', methods=['get','POST'])
def predict():

    # Get data from request form
    data = request.form['user_input']

    # Convert data from unicode to string
    data = str(data)

    # If nothing was entered, send error message to user and don't pass to model
    if len(data) == 0:
        pred = "you didn't enter anything"
        suggest = "brush yourself off and try again"

    # Make prediction based on new data
    else:
        top_genre = pred_genre(data)
        pred = 'Sounds like some sweet sweet {} to me!'.format(top_genre)

        new_genre, replacement = make_suggestion(top_genre)
        suggest = 'Want to make it a little more {}? Try adding "{}"'.format( \
            new_genre, replacement)

    # Return a string format of that prediction to the html page
    return render_template('index.html', pred=pred, suggest=suggest, last=data)


def make_suggestion(genre):
    """
    INPUT: string -- predicted genre of user input
    OUTPUT: string -- random new genre (other than predicted one)
            string -- random important word from new genre

    Takes in the predicted genre of the user text and suggests a word that could
    be used to alter the text enough to change the prediction to a different
    randomly selected genre.

    """

    # Get a list of all of the genres the model knows about
    all_genres = list(m.classes_)

    # Take off the currently predicted genre from the list
    all_genres.remove(genre)

    # Randomly select a new genre from the list
    new_genre = np.random.choice(all_genres)

    # Get a ilst of the top 26 'defining' words of the new genre
    word_options = key_words_dict[new_genre]

    # Randomly select one of the them
    replacement = np.random.choice(word_options)

    # Return the new genre and replacement word
    return new_genre, replacement


def transform_text(string, cutoff=100):
    """
    INPUT: string -- text to be transformed to vector
           int -- point in [0-5000] at which to cut off commmon/stopwords
    OUTPUT: array -- vector representing normalized word count for top 5k words

    Takes in a string and vectorizes it. Ignores X most common words in vector
    construction where X is defined by the value of cutoff.
    """

    # Catch a few common unicode problem characters
    string = string.replace("\xe2\x80\x99","'")
    string = string.replace("\xe2\x80\xa6","...")
    string = string.replace("\u2018","")
    string = string.replace("\u2019","")

    # Tokenize string
    string = tokenize(string.strip(punctuation))

    # Create empty array to hold counts of each of top 5k words
    to_fill = np.zeros(5001)


    for word in string:
        # Ignore words not in top 5000
        if word in top5k_dict:
            # Only use word if it's beyond cutoff value
            if top5k_dict[word] >= cutoff:
                # Increment count of word in array
                to_fill[top5k_dict[word]] += 1

    # Normalize counts using total number of words in input string
    to_fill /= len(string)*1.0

    return to_fill


def pred_genre(string):
    """
    INPUT: string -- input text from user
    OUTPUT: string -- genre predicted by model
    """
    # Convert input string to vector format
    vector = transform_text(string)
    vector = vector.reshape(1, -1)

    # Get a list of class/genre indicies sorted by probability for input string
    prob_list = np.argsort(m.predict_proba(vector))

    # Using predicted class order generate class/genre name as string
    top_genre = m.classes_[prob_list][0][::-1][0]

    return top_genre


def tokenize(doc):
    """
    INPUT: string
    OUTPUT: list of stemmed words from string

    A simple tokenizing function.
    """
    return [snowball.stem(word) for word in word_tokenize(doc.lower())]


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
