
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from string import punctuation



def load_genres(filename='../jawnre_extra/msd_tagtraum_cd2_fix.txt'):
    """
    INPUT: txt file with MSD trackIDs and their Majority and Minority genres
    OUTPUT: pandas dataframe with same information

    Opens file with song genres classification and returns pandas database with
    row for each song.

    SOURCE FILE URL: http://www.tagtraum.com/genres/msd_tagtraum_cd2.cls.zip
    Open file beforehand in excel or similar for columns to properly register.
    """

    # Pandas will read file as having five columns so create names for each
    col_names = ['trackID', 'Majority', 'Minority', 'Extra1', 'Extra2']

    # File is tab separated with comment lines marked with '#' and no header
    data = pd.read_csv(filename, delimiter='\t', comment='#', header=None,
                       names = col_names)

    # Drop the extra two blank columns
    data.drop(['Extra1', 'Extra2'], axis = 1, inplace=True)
    data.drop(data.index[[0]], inplace=True)

    return data


def load_misc_files():
    """
    INPUT: none
    OUTPUT: various list and dictionary versions of the song lyrics

    Opens the four files that contain different information on song lyrics and
    return them as lists and dictionaries for pickling.

    SOURCE FILE URLS:
    http://labrosa.ee.columbia.edu/millionsong/sites/default/files/
        AdditionalFiles/full_word_list.txt.zip
        mxm_reverse_mapping.txt
        AdditionalFiles/mxm_dataset_test.txt.zip
        AdditionalFiles/mxm_dataset_train.txt.zip
    """

    # Open file of top 5000 stemmed words and store as list
    # Note that list will be zero indexed and represents the full vocabulary
    with open('../jawnre_extra/top5000.txt') as f:
        top5k_list = f.read().split(',')

    # Convert top 5000 list to lookup dictionary (key=stemmed word, value=index)
    # Note that this dict will be 'one indexed' to align with future vectors
    top5k_dict = {}
    for index, word in enumerate(top5k_list):
        top5k_dict[word] = index + 1

    # Open file with reverse stemming map (stemmed --> unstemmed)
    with open('../jawnre_extra/mxm_reverse_mapping.txt') as f:
        reverse_map_lines = f.readlines()
    # Create a reverese lookup dictionary to convert to unstemmed when needed
    rev_map_dict = {}
    for line in reverse_map_lines:
        line = line.split('<SEP>')
        rev_map_dict[line[0]] = line[1].rstrip()

    # Full bag of words data split into train and test files
    # Open test songs bag of words data and store as list of lines
    with open('../jawnre_extra/mxm_dataset_test.txt') as f:
        Ltest = f.readlines()

    # Open train songs bag of words data and store as list of lines
    with open('../jawnre_extra/mxm_dataset_train.txt') as f:
        Ltrain = f.readlines()

    return top5k_list, top5k_dict, rev_map_dict, Ltest, Ltrain


def ID2BoW_map(lines):
    """
    INPUT: list of lines (read from lyrics files)
    OUTPUT: dictionary

    Creates a lookup dictionary where key = songID, value = list of bag of words
    for that song. Each item in list is in format word_index:word_count
    """
    dict_of_songs = {}
    for line in lines:
        if not line.startswith('#') and not line.startswith('%'):
        # if line begins with '#' or '%' is is not valid data
            line = line.rstrip()
            # convert eahc line from string to list
            line_list = line.split(',')
            # create entry in dict where key = songID, value = BoW list
            dict_of_songs[line_list[0]] = line_list[2:]

    return dict_of_songs


def match_BoW_to_genre(mapping, genres, cutoff=50):
    """
    INPUT: dictionary -- mapping between ID and BoW,
           dataframe -- mapping between ID and genre,
           int -- cutoff point for most popular words (ie exclude top X words)
    OUTPUT: list of arrays -- each entry is numeric vector repsenting word count
            list -- sibling list (same order/length) with targets (genres)

    Takes in raw BoW and genre data and connects and prepares it for use in
    model training.

    Combines ID to BoW mapping with ID to genre mapping and finds songs
    that exist in both maps. For those songs the function then converts BoW from
    list format to array format and creates an entry in export list with that
    array as well as creating an entry in export list of genres with that
    vector's target.

    Songs belonging to a genre that was chosen to be excluded (see project
    documentation for more info) are ignored. Word counts for words that are in
    the top X words (ie stopwords, as defined by the cutoff variable) are set to
    zero.
    """

    genres_to_exclude = ['Latin', 'World', 'Jazz', 'New Age', 'Pop', 'Rock']
    ordered_vects = []
    ordered_genres = []

    for song in genres.values:
    # For each of the 280K tagtraum genre entries (rows)

        songID = song[0]
        song_genre = song[1]

        if songID in mapping and song_genre not in genres_to_exclude:
        # If that song's ID is in the MXM lyrics dict as a key AND if that
        # song's genre is not on the exclude list then...

            # Create empty array for the song to hold counts of top 5000 words
            to_fill = np.zeros(5001)

            # Look up the song's BoW (still as a list)
            bow_list = mapping[songID]

            # Convert list of word counts to array format
            for index_and_count in bow_list:
                # Each entry in format [X:Y], where X = word index on master 5k list, Y = count
                index_and_count = index_and_count.split(':')
                # Get the index of the word on master 5k list
                word_index = int(index_and_count[0])
                # Get the count of that word in the song
                word_count = int(index_and_count[1])

                # Only enter word's count if its below stopwords cutoff point
                if word_index >= cutoff:
                    count = word_count
                else:
                    count = 0
                # Insert word count into appropriate index in vector
                to_fill[word_index] = count

            # Put finished array for that song into list
            ordered_vects.append(to_fill)
            # Put genre of that song into list
            ordered_genres.append(song_genre)

    # Returns two lists of equal lenth in same order:
    #   ordered_vects -- list of word count vectors for songs
    #   ordered_genres -- list of genres/labels for those songs
    return ordered_vects, ordered_genres


def build_model(data, labels):
    """
    INPUT: list -- of arrays representing word counts for each song
           list -- of labels (as strings)
    OUTPUT: multinomial naive bayes model trained with input data and labels

    Creates and fits a MNB model, returns the fit model for the given data
    """

    # Uniform priors set to avoid problems with class imbalance in training data
    # that don't translate when classifying non-musical text
    multiNB = MultinomialNB(fit_prior=False)

    fitted_model = multiNB.fit(data, labels)

    return fitted_model


def calc_key_words(genre_num):
    """
    INPUT: int -- index of genre from .classes_ of fit_model
    OUTPUT: string -- genre name
            list -- "key" words for the genre

    Returns a list of 25 words for the genre that much more common in it than in
    the others. (ie the words that help "define" it) List later used to provide
    suggestions on how to transform a string from one genre to another.
    """

    # Get the genre name from the fit model
    genre_name = m.classes_[genre_num]

    # Create an empty array to hold word counts (similar to for songs elsewhere)
    total = np.zeros(5001)

    # Sum up the word counts for all genres other than input genre
    for index, f_count in enumerate(m.feature_count_):
        if index != genre_num:
            # Get average count of each word per song in a genre
            normal_count = m.feature_count_[index] / m.class_count_[index]

            # Add those counts to the running total
            total += normal_count

    # Calculate average word counts per song for genres other than input genre
    total_norm = total/(len(m.classes_)-1)

    # Get average count of each word per song in input genre
    input_normal_count = m.feature_count_[genre_num] / m.class_count_[genre_num]

    # Calculate how many times more than the (non-input) mean each word appears
    # in the input genre
    feat_above_mean = input_normal_count / (total_norm + .00005)
    # Convert any NaNs resulting from above to zeros
    feat_above_mean = np.nan_to_num(feat_above_mean)

    # Generate sorted list of word indicies (in order of 'importance' to genre)
    top_words_this_genre = np.argsort(feat_above_mean)[::-1]

    cleaned_list = clean_top_words(top_words_this_genre, total_norm)

    return genre_name, cleaned_list

def clean_top_words(top_words, total_norm):
    """
    INPUT: array -- list of word indicies sorted by importance to their genre
           array -- average word counts per song for genres other than this one
    OUTPUT: list -- cleaned list of the top 25 of those words

    Takes in full list of sorted word indicies and checks if each word in order
    is on the blacklist of non-english/badly formatted/NSFW words and returns a
    clean list of the top 25.
    """

    list_of_key_words = []

    counter = 0
    for word in top_words[0:150]:
        # Get the stemmed version of the word as a string from lookup dict
        stem_word = top5k[word-1].rstrip()
        # Get an unstemmed version of the word from another lookup dict
        real_word = rev_map_dict[stem_word]

        # Get the mean freq of this word in other genres
        mean_freq_elsewhere = total_norm[word]

        if mean_freq_elsewhere > .001 and real_word not in BLACKLIST and \
           counter <= 25:
                counter += 1
                list_of_key_words.append(real_word)

    return list_of_key_words


def create_important_dict():
    """
    INPUT: none
    OUTPUT: dict

    Iterates through all genres to make dictionary where keys are genre names,
    values are list of important words for that genre.
    """
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

    BLACKLIST = ['od', 'za', 'worte', 'tod', 'lai', 'eg', 'toma', 'tener', \
                'quién', 'kalt', 'nana', 'gott', 'herz', 'blut', 'licht', \
                'nein', 'seele', 'i´m', 'it´s', 'weißt', 'tief' , 'tá', 'när', \
                'fuori', 'occhi' , 'guarda', 'sotto', 'Â', 'alleluia!', 'ref:',\
                'meg', 'jeg', 'don`t', 're', 'dot', 'saa', 'nacht', "(x4)", \
                '(x3)', 'himmel', 'gesicht', 'traum', 'don´t', 'niemand', \
                'mina', 'note', 'niet', 'ik', 'allt' , 'senza', 'ven', 'ke', \
                'yi', 'junto', 'hermano', 'digo', 'pum', 'leve', 'ann', 'sä', \
                'dort' ,'meinem', 'vem', "qu'", 'yer', 'ar', 'bara', 'inte', \
                'notte', 'finns', "i’ve", 'lei', 'här', 'att', 'suo', 'ora', \
                'gli', 'då', 'aldrig', 'delle', 'får', 'dal', 'för', 'giorno', \
                'och' , 'cunt', 'asshole', 'carol', 'niggaz', "motherfuckin'",\
                'hoes', 'bitch', 'shit', 'motherfucker', 'ikke', '50', 'p', \
                "t'as", 'ass', 'contra', 'pueblo', 'bam', 'dong', 'gi', 'di', \
                'mi', 'fi', 'fe', 'poder', 'mujer', 'queda', 'nena', 'dale',\
                'nombre', 'vos', 'muy', 'einmal', 'tierra', "l'on", 'esa', \
                'piel', 'cuerpo', 'nuestro', 'mentira', 'libre', 'fuego', 'ka',\
                'u', 'dah', 'christmas', 'claus', 'ba', 'ow', 'merry', 'bop', \
                'll']

    key_words_dict = create_important_dict()

    with open('model.pkl', 'w') as f:
        pickle.dump(m, f)

    with open('lookups.pkl', 'w') as f:
        pickle.dump([top5k, top5k_dict, rev_map_dict, key_words_dict], f)
