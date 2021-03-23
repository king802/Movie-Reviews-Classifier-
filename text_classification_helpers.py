import glob
import string

import nltk
import numpy as np
import pandas as pd

nltk.download('punkt')
from nltk.tokenize import word_tokenize


def read_data(path: str, labeled: bool = True):
    """
    Takes in a file path of which contains 2 sub folders (neg and pos)
    :param path: the path from current directory to the directory where reviews are located.
    :param labeled: Boolean telling function if the reviews have labels or not. Defaults to true
    :return: A list that contains all of the files information
            format => [[filename, text, label(optional)] ...]
    """

    def get_file_name(fullPath: str):
        """
        grabs the filename of the path that is given.
        :param fullPath: path to the file that's name is being retrieved.
        :return: The name of the file.
        """
        return fullPath.split("/")[-1].split(".")[0]

    def get_reviews_info(filePath: str, label=None):
        """
        Grabs the relevant information about the review given the filePath of the review.
        :param filePath: the path to the review.
        :param label: the label of the file. 1 for pos, 0 for neg and None if no label (default)
        :return: a tuple of all the reviews data => (fileName, Review Text, Label)
        """
        if label is None:
            with open(filePath, 'r') as file:
                review = file.read()
            return (get_file_name(filePath), review)
        else:
            with open(filePath, 'r') as file:
                review = file.read()
            return (get_file_name(filePath), review, label)

    if labeled:
        pos_review_paths = glob.glob(path + "/pos/*")
        neg_reviews_paths = glob.glob(path + "/neg/*")
        pos_reviews = list(map(lambda x: get_reviews_info(x, 1), pos_review_paths))
        neg_reviews = list(map(lambda x: get_reviews_info(x, 0), neg_reviews_paths))
        return pos_reviews + neg_reviews
    else:
        reviews_paths = glob.glob(path + "/*")
        reviews = list(map(get_reviews_info, reviews_paths))
        return reviews


def create_vocabulary(data, b: "The Cut off for words that are to infrequent" = 5):
    """
    Builds out a vocabulary from all of the data that is given to it. This vocabulary wont have words that appear
    less then 'b' times in the given data.
    :param data: the reviews data [(fileName, Review, Label) ...].
    :param b: the cutoff threshold for words that shouldn't be in the vocabulary. Default to 5.
    :return: the vocabulary of all the words in the data set provided that appear more frequently then b.
    """
    vocab = {}
    stop = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they",
            "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
            "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
            "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at",
            "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
            "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
            "can", "will", "just", "don", "should", "now"] + list(string.punctuation)

    for review in data:
        words = word_tokenize((review[1]))
        for word in words:
            if word in vocab.keys():
                vocab[word] = vocab[word] + 1
            else:
                vocab[word] = 1
    tokens_to_remove = []
    for word in vocab.keys():
        if vocab[word] < b:
            tokens_to_remove.append(word)
    for word in tokens_to_remove + stop:
        vocab.pop(word, None)
    return vocab


def transform(document: str, vocab: "The vocab built out from the reviews"):
    """
    Given a specific review it will convert its text into a bag of words vector.
    :param document: the text of a given review.
    :param vocab: the vocab built out for the data set.
    :return: a vector of the how many times each vocab word appears in the text. [ 10, 2, 0, 0 ....].
    """
    vector = [0] * len(vocab)
    for i, feature in enumerate(vocab):
        vector[i] = document.count(" " + feature + " ")
    return vector


def make_data_frame(reviews: "List of the reviews in format [(id, text, label)]",
                    vocab: "The vocab built out from the reviews"):
    """
    This function takes in a list of the reviews [(id, text, label)] and then converts it to bag of words labels and
    puts it in pandas dataframe.
    :param reviews: the list of reviews to be converted.
    :param vocab: the vocab built out from the reviews.
    :return: a data frame that has all of the reviews as rows and the columns are Id, vocab keys and label.
    """
    columns = ['Id'] + list(vocab.keys()) + ['Label']
    rows = list(map(lambda x: [x[0]] + transform(x[1], vocab) + [x[2]], reviews))
    df = pd.DataFrame(rows, columns=columns)
    return df


def predict(text, model):
    """
    Given a text and a probabilistic model it returns the classification of the reviews.
    :param text: the text to be classified.
    :param model: the model to be used to predict the
    :return: the predicted label of the text that was given.
    """
    percentage_pos = np.log(model['NBTRUEVALUE'])
    percentage_neg = np.log(model["NBFALSEVALUE"])
    for word in text.split():
        if word in model.keys():
            prob_pos, prob_neg = model[word]
            percentage_pos += np.log(prob_pos)
            percentage_neg += np.log(prob_neg)
    if percentage_pos > percentage_neg:
        return 1
    else:
        return 0


def make_probability_distribution(df, alpha=1):
    """
    This function takes in a dataframe and returns a probability distribution of how each word in the vocabulary
    dictates the review.
    :param df: The dataset that is to be used to build out predictive model.
    :param alpha: is the smoothing factor that can be applied to the model (default = 1).
    :return: a probabilistic model that tells what the probability given a word from the vocab in dictating a positve
    or negative review. Format => { feature: (probability of being positive review, probability of negative review),
    ...}
    """
    vocab = list(df.keys())
    vocab.pop(0)
    vocab.pop(-1)
    positive_data = df[df['Label'] == 1]
    negative_data = df[df['Label'] == 0]
    vocab_size = len(vocab)
    total_features_pos = 0
    total_features_neg = 0
    vector_space = {'NBTRUEVALUE': sum(df['Label']) / len(df['Label']),
                    "NBFALSEVALUE": 1 - sum(df['Label']) / len(df['Label'])}

    for feature in vocab:
        total_features_pos += sum(positive_data[feature])
        total_features_neg += sum(negative_data[feature])
    for feature in vocab:
        N_tp = sum(positive_data[feature])
        N_tn = sum(negative_data[feature])
        vector_space[feature] = ((N_tp + alpha) / (total_features_pos + (alpha * vocab_size)),
                                 (N_tn + alpha) / (total_features_neg + (alpha * vocab_size)))
    return vector_space


def evaluate(dataset, model):
    """
    This function evaluates how well done our prediction model was.
    :param dataset: the full dataset with its labels that was predicted on.  
    :param model: the model that was used to predict the reviews. 
    :return: metrics for how well model did (Accuracy, TP, TN, FP, FN)
    """
    true_positives = 0  # predicted it to be 1 and it was 1
    true_negatives = 0  # predicted it to be 0 and it was 0
    false_positives = 0  # predicted to be 1 but should be 0
    false_negatives = 0  # predicted to be 0 but should be 1
    for id, text, label in dataset:
        predicted = predict(text, model)
        if predicted == label:
            if label == 0:
                true_negatives += 1
            else:
                true_positives += 1
        else:
            if label == 0:
                false_positives += 1
            else:
                false_negatives += 1
    accuracy = (true_positives + true_negatives) / len(dataset)
    return "Accuracy: " + str(accuracy) + "\nTP: " + str(true_positives) + "\nTN: " + str(true_negatives) + "\nFP: " \
           + str(false_positives) + "\nFN: " + str(false_negatives)


def predict_values(text, model, id):
    """
    This function returns the resulats from the prediction for a given review.
    :param text: the review text
    :param model: the mdoel to predict with.
    :param id: the id of the review.
    :return: the probability distribution of the review P(+), P(-)
    """
    p_pos = np.log(model['NBTRUEVALUE'])  # Percentage of positive reviews
    p_neg = np.log(model["NBFALSEVALUE"])  # Percentage of negative reviews

    for word in text.split():
        if word in model.keys():
            p_pos += np.log(model[word][0])
            p_neg += np.log(model[word][1])
    return str(id) + " | \nP(+): log(" + str(p_pos) + ")\nP(-): log(" + str(p_neg) + ")"
