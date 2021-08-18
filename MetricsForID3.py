from functions import get_word_appearances, get_vocabulary, get_reviews_vectors
from ID3 import ID3
import numpy as np
import matplotlib.pyplot as plt

def train(available_words, train_positive_reviews, train_negative_reviews, train_reviews):
    """
    :param available_words: A dictionary (key: index, value: word) with keys representing the unique indexes of the words 
                            and values being the words in the file.
    :param reviews: Contains all the reviews from the labeledBow.feat file.
    :param negative_reviews: Contains only the negative reviews from the train/labeledBow.feat file.
    :param positive_reviews: Contains only the positive reviews from the train/labeledBow.feat file.
    :return: vocabulary: A dictionary (key: index, value: information_gain) with keys representing the unique indexes of the 
                         words and values being the return values of the information_gain() function for each specific index
                         (word).
             vocabulary2: A numpy array that contains only the words from the vocabulary(Basically, it contains the indexes of 
                          the words, like they are stored in imdb.vocab file)
             vectored_reviews: A 2d numpy array with shape ( len(reviews), len(vocabulary)+1 ) where each row represents one 
                               review. Every review is a binary numpy array. The first item indicates the category of the 
                               review and the following items whether or not this review contains a specific index (word).
             len(negative_reviews): The length of the negative reviews
             len(positive_reviews): The length of the positive reviews
    """
    
    word_appearances = get_word_appearances(train_reviews, available_words)
    negative_word_appearances = get_word_appearances(train_negative_reviews, available_words)
    positive_word_appearances = get_word_appearances(train_positive_reviews, available_words)

    vocabulary = get_vocabulary([len(train_negative_reviews), len(train_positive_reviews)], [negative_word_appearances, positive_word_appearances], word_appearances, 26)

    vocabulary2 = []
    for word in vocabulary.keys():
        vocabulary2.append(word)
    vocabulary2 = np.array(vocabulary2)

    vectored_reviews = np.array(get_reviews_vectors(train_reviews, vocabulary))

    return vocabulary2, vectored_reviews, len(train_negative_reviews), len(train_positive_reviews), vocabulary


def classifier(root, vector, vectored_reviews):
    """
    :param root: The root of the tree, which tree is result from the train data.
    :param vector: The vector with the most valuable words (we use it as a counter).
    :param vectored_reviews:  A 2d numpy array with shape ( len(reviews), len(vocabulary)+1 ) where each row represents one 
                              review. Every review is a binary numpy array. The first item indicates the category of the 
                              review and the following items whether or not this review contains a specific index (word).
    :return: correct_positives: A number, which contains the summary of the positive reviews that they have been classified correctly.
             correct_negatives: A number, which contains the summary of the negative reviews that they have been classified correctly.
             false_positives: A number, which contains the summary of the negative reviews that they have been classified as positive wrongly.
             false_negatives: A number, which contains the summary of the positive reviews that they have been classified as negative wrongly.
    """
    
    categories = []
    correct_positives = 0
    correct_negatives = 0
    false_positives = 0
    false_negatives = 0
    for review in vectored_reviews:
        current_node = root
        i = 1
        while current_node is not None:
            default_category = current_node.default_category
            if i < len(vector):
                if review[i] == 1:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            else:
                break
            i += 1
        categories.append(default_category)
        if review[0] == 1 and default_category == "positive":
            correct_positives += 1
        elif review[0] == 0 and default_category == "positive":
            false_positives += 1
        elif review[0] == 0 and default_category == "negative":
            correct_negatives += 1
        elif review[0] == 1 and default_category == "negative":
            false_negatives += 1
    pos_counter = 0
    neg_counter = 0
    for cat in categories:
        if cat == "negative":
            neg_counter += 1
        else:
            pos_counter += 1
    
    accuracy = 100 * (correct_negatives + correct_positives) / len(vectored_reviews)
    print("Positive and Negative counters", pos_counter, neg_counter)
    print("Correct positives: ", correct_positives)
    print("Correct negatives: ", correct_negatives)
    print("False positives: ", false_positives)
    print("False negatives: ", false_negatives)
    print("Accuracy = ", accuracy, "%")
    return correct_positives, correct_negatives, false_positives, false_negatives

def get_accuracy(correct_positives, correct_negatives, false_positives, false_negatives):
    """
    :param correct_positives: A number, which contains the summary of the positive reviews that they have been classified correctly.
    :param correct_negatives: A number, which contains the summary of the negative reviews that they have been classified correctly.
    :param false_positives: A number, which contains the summary of the negative reviews that they have been classified as positive wrongly.
    :param false_negatives: A number, which contains the summary of the positive reviews that they have been classified as negative wrongly.
    :return: accuracy: which is equal to (correct_positives + correct_negatives) / (correct_positives + correct_negatives + false_positives + false_negatives)
    """
    accuracy = (correct_positives + correct_negatives) / (correct_positives + correct_negatives + false_positives + false_negatives)
    return accuracy


def get_error(correct_positives, correct_negatives, false_positives, false_negatives):
    """
    :param correct_positives: A number, which contains the summary of the positive reviews that they have been classified correctly.
    :param correct_negatives: A number, which contains the summary of the negative reviews that they have been classified correctly.
    :param false_positives: A number, which contains the summary of the negative reviews that they have been classified as positive wrongly.
    :param false_negatives: A number, which contains the summary of the positive reviews that they have been classified as negative wrongly.
    :return: error: which is equal to (false_positives + false_negatives) / (correct_positives + correct_negatives + false_positives + false_negatives)
    """
    error = (false_positives + false_negatives) / (correct_positives + correct_negatives + false_positives + false_negatives)
    return error


def get_learning_curve_points(train_negative_reviews, train_positive_reviews, train_reviews, test_reviews, available_words):
    """
    :param train_negative_reviews: A list with the train negative reviews 
    :param train_positive_reviews: A list with the train positive reviews
    :param train_reviews: A list with the train reviews
    :param test_reviews: A list with the test reviews
    :param available words: A dictionary (key: index, value: word) with keys representing the unique indexes of the words 
                            and values being the words in the file.
    :return: x_values: a list which contains the values of x axis
             train_y_values: a list which contains the train values of y axis
             test_y_values: a list which contains the test values of y axis
    """
    x_values = []
    train_y_values = []
    test_y_values = []
    
    for value in np.linspace(0.1, 1.0, num=10):
        x_values.append(value)

        used_train_negative_reviews = train_negative_reviews[:int(round(len(train_negative_reviews)*value))]
        used_train_positive_reviews = train_positive_reviews[:int(round(len(train_positive_reviews)*value))]
        used_train_reviews = used_train_negative_reviews + used_train_positive_reviews
        
        vector, train_data, neg_length, pos_length, vocabulary = \
        train(available_words, used_train_positive_reviews, used_train_negative_reviews, used_train_reviews)
        m = 'positive'
        tree = ID3()
        root = tree.insert(vector, train_data, neg_length, pos_length, m, 1)
        
        vectored_reviews = np.array(get_reviews_vectors(used_train_reviews, vocabulary))
        train_true_positives, train_true_negatives, train_false_positives, train_false_negatives = classifier(root, vector, vectored_reviews)
        train_accuracy = get_accuracy(train_true_positives, train_true_negatives, train_false_positives, train_false_negatives)
        train_y_values.append(train_accuracy)
        
        test_vectored_reviews = np.array(get_reviews_vectors(test_reviews, vocabulary))
        test_true_positives, test_true_negatives, test_false_positives, test_false_negatives = classifier(root, vector, test_vectored_reviews)
        test_accuracy = get_accuracy(test_true_positives, test_true_negatives, test_false_positives, test_false_negatives)
        test_y_values.append(test_accuracy)

    return x_values, train_y_values, test_y_values


def plot_learning_curve(x_values, train_y_values, test_y_values):
    """
    :param x_values: a list which contains the values of x axis
    :param train_y_values: a list which contains the train values of y axis
    :param test_y_values: a list which contains the test values of y axis
    """
    fig = plt.figure(figsize=(10, 6))

    fig.suptitle("Learning Curves")

    acc_chart = fig.add_subplot(121)
    acc_chart.plot(x_values, train_y_values, color="k")
    acc_chart.plot(x_values, test_y_values, color="r")
    acc_chart.set_xlabel("Percentage of data used.")
    acc_chart.set_ylabel("Accuracy")
    acc_chart.legend(["Train Data", "Test Data"])

    error_chart = fig.add_subplot(122)
    error_chart.plot(x_values, [1-acc for acc in train_y_values], color="k")
    error_chart.plot(x_values, [1-acc for acc in test_y_values], color="r")
    error_chart.set_xlabel("Percentage of data used.")
    error_chart.set_ylabel("Error")
    error_chart.legend(["Train Data", "Test Data"])

    plt.show()
