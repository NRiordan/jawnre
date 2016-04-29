
# coding: utf-8

import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from string import punctuation
import pickle


def load_genres(filename='../jawnre_extra/msd_tagtraum_cd2_fix.txt'):
    # load all trackIDs with genre labels into dataframe and return it
    # NOTE: open file once beforehand in excel or similar for columns to properly register
    col_names = ['trackID', 'Majority', 'Minority', 'Extra1', 'Extra2']
    data = pd.read_csv(filename, delimiter='\t', comment='#', header=None, names = col_names)
    data.drop(['Extra1', 'Extra2'], axis = 1, inplace=True)
    data.drop(data.index[[0]], inplace=True)
    return data

def load_misc_files():

    # open list of top 5000 stemmed words and store as list
    # LIST IS ZERO INDEXED!
    # BoW IS ONE INDEXED!
    with open('../jawnre_extra/top5000.txt') as f:
        top5k_list = f.read().split(',')

    # open reverse mapping file (stemmed --> unstemmed) and create lookup dict
    with open('../jawnre_extra/mxm_reverse_mapping.txt') as f:
        reverse_map_lines = f.readlines()

    rev_map_dict = {}
    for line in reverse_map_lines:
        line = line.split('<SEP>')
        rev_map_dict[line[0]] = line[1].rstrip()

    # open test list of songs and store as list of lines
    with open('../jawnre_extra/mxm_dataset_test.txt') as f:
        Ltest = f.readlines()

    # open train list of songs and store as list of lines
    with open('../jawnre_extra/mxm_dataset_train.txt') as f:
        Ltrain = f.readlines()

    # convert list to lookup dict for input text
    top5k_dict = {}
    for index, word in enumerate(top5k_list):
        top5k_dict[word] = index + 1

    return top5k_list, top5k_dict, rev_map_dict, Ltest, Ltrain

def build_model(data, targets):
    # creates and fits a MNB model, returns the fit model for the given data

    multiNB = MultinomialNB(fit_prior=False)
    fitted = multiNB.fit(data, targets)

    return fitted

def ID2BoW_map(lines):
    dict_of_songs = {}
    for line in lines:
        if not line.startswith('#') and not line.startswith('%'):
            line = line.rstrip()
            line_list = line.split(',')
            #create entry in dict where key = songID, value = BoW list
            dict_of_songs[line_list[0]] = line_list[2:]
    return dict_of_songs

def match_BoW_to_genre(mapping, genres, cutoff=50, to_exclude=['Latin', 'World', 'Jazz', 'New Age', 'Pop', 'Rock']):

    #NOTE: maybe use alternate genre for rock assignments if time, for now exclude

    ordered_vects = []
    ordered_genres = []
    for song in genres.values:
    # for each of the 280K tagtraum genre entries

        if song[0] in mapping and song[1] not in to_exclude:
        #if it's ID (position 0) is in the MXM lyrics dict as key

            to_fill = np.zeros(5001)

            # lookup its BoW (still as list)
            bow_list = mapping[song[0]]

            # convert list of word counts to array format
            for vect in bow_list:
                # vect in format [X:Y], where X = word index on master 5k list, Y = count
                vect = vect.split(':')
                # get the index of the word on master 5k list
                word_index = int(vect[0])

                # only enter word if its below cutoff on list (ie exclude top X most used words)
                if word_index >= cutoff:
                    count = int(vect[1])
                else:
                    count = 0
                to_fill[word_index] = count

            # put finished array into list
            ordered_vects.append(to_fill)
            # song genre in postion 1
            ordered_genres.append(song[1])

    # returns two lists of equal lenth in same order:
    # ordered_vects : list of BoW
    # ordered_genres : list of genres/labels

    return ordered_vects, ordered_genres


def calc_key_words(genre_num):
    list_of_key_words = []
    genre_name = m.classes_[genre_num]
    total = np.zeros(5001)

    for index, f_count in enumerate(m.feature_count_):
        if index != genre_num:
            # get average count of word per song in that genre
            normal_count = m.feature_count_[index] / m.class_count_[index]

            total += normal_count

    # average count for genres other than the one being calc'd
    total_norm = total/(len(m.classes_)-1)

    feat_above_mean = (m.feature_count_[genre_num]/m.class_count_[genre_num]) / (total_norm + .00005)
    feat_above_mean = np.nan_to_num(feat_above_mean)

    top_words_this_genre = np.argsort(feat_above_mean)[::-1]

    counter = 0
    for word in top_words_this_genre[0:150]:
        real_word = rev_map_dict[top5k[word-1].rstrip()]
        times_above_mean = round(feat_above_mean[word], 2)
        mean_freq_elsewhere = round(total_norm[word], 5)

        if mean_freq_elsewhere > .001 and real_word not in blacklist and counter <= 25:
            counter += 1
            list_of_key_words.append(real_word)

    return genre_name, list_of_key_words


blacklist = ['od', 'za', 'worte', 'tod', 'lai', 'eg', 'toma', 'tener', 'quién', 'kalt', 'nana', 'gott', 'herz', \
            'blut', 'licht', 'nein', 'seele', 'i´m', 'it´s', 'weißt', 'tief' , 'tá', 'när', 'fuori', 'occhi' ,\
            'guarda', 'sotto', 'Â', 'alleluia!', 'ref:', 'meg', 'jeg', 'don`t', 're', 'dot', 'saa', 'nacht', "(x4)", \
            '(x3)', 'himmel', 'gesicht', 'traum', 'don´t', 'niemand', 'mina', 'note', 'niet', 'ik', 'allt' ,\
            'senza', 'ven', 'ke', 'yi', 'junto', 'hermano', 'digo', 'pum', 'leve', 'ann', 'sä', 'dort' ,\
            'meinem', 'vem', "qu'", 'yer', 'ar', 'bara', 'inte', 'notte', 'finns', "i’ve", 'lei', 'här', \
            'att', 'suo', 'ora', 'gli', 'då', 'aldrig', 'delle', 'får', 'dal', 'för', 'giorno', 'och' ,\
            'cunt', 'asshole', 'carol', 'niggaz', "motherfuckin'", 'hoes', 'bitch', 'shit', 'motherfucker', \
            'ikke', '50', 'p', "t'as", 'ass', 'contra', 'pueblo', 'bam', 'dong', 'gi', 'di', 'mi', 'fi', 'fe',\
            'poder', 'mujer', 'queda', 'nena', 'dale', 'nombre', 'vos', 'muy', 'einmal', 'tierra', "l'on", \
            'esa', 'piel', 'cuerpo', 'nuestro', 'mentira', 'libre', 'fuego', 'ka', 'u', 'dah', 'christmas', \
            'claus', 'ba', 'ow', 'merry', 'bop', 'll']

def create_important_dict():
    impt_dict = {}
    for genre_num in range(len(m.classes_)):
        genre_name, top_words = calc_key_words(genre_num)
        impt_dict[genre_name] = top_words
    return impt_dict



if __name__ == '__main__':

    genres = load_genres()
    top5k, top5k_dict, rev_map_dict, Ltest, Ltrain = load_misc_files()
    test_map = ID2BoW_map(Ltest)
    train_map = ID2BoW_map(Ltrain)

    # combine train + test dicts to form mapping for full dataset
    full_map = train_map.copy()
    full_map.update(test_map)

    X_full, y_full = match_BoW_to_genre(full_map, genres)

    # create and fit model for full dataset
    m = build_model(X_full, y_full)

    key_words_dict = create_important_dict()

    with open('model.pkl', 'w') as f:
        pickle.dump(m, f)

    with open('lookups.pkl', 'w') as f:
        pickle.dump([top5k, top5k_dict, rev_map_dict, key_words_dict], f)
