import functions
import naiveBayes
import numpy as np
import matplotlib.pyplot as plt


def split_train_dev(negative_reviews, positive_reviews):

    pos_percent_95 = round(len(positive_reviews) * 0.95)
    neg_percent_95 = round(len(negative_reviews) * 0.95)

    train_positive_reviews = positive_reviews[:pos_percent_95]
    train_negative_reviews = negative_reviews[:neg_percent_95]
    train_reviews = train_positive_reviews + train_negative_reviews

    dev_reviews = negative_reviews[neg_percent_95:] + positive_reviews[pos_percent_95:]

    return train_negative_reviews, train_positive_reviews, train_reviews, dev_reviews


def get_accuracy(true_positives, true_negatives, false_positives, false_negatives):
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    return accuracy


def get_error(true_positives, true_negatives, false_positives, false_negatives):
    error = (false_positives + false_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    return error


def get_learning_curve_points(train_negative_reviews, train_positive_reviews, test_reviews, available_words):
    x_values = []
    train_y_values = []
    test_y_values = []

    for value in np.linspace(0.1, 1.0, num=10):
        x_values.append(value)

        used_train_negative_reviews = train_negative_reviews[:int(round(len(train_negative_reviews)*value))]
        used_train_positive_reviews = train_positive_reviews[:int(round(len(train_positive_reviews)*value))]
        used_train_reviews = used_train_negative_reviews + used_train_positive_reviews

        word_appearances = functions.get_word_appearances(used_train_reviews, available_words)
        negative_word_appearances = functions.get_word_appearances(used_train_negative_reviews, available_words)
        positive_word_appearances = functions.get_word_appearances(used_train_positive_reviews, available_words)

        vocabulary = functions.get_vocabulary([len(used_train_negative_reviews), len(used_train_positive_reviews)],
                                              [negative_word_appearances, positive_word_appearances], word_appearances,
                                              1200)

        negative_vectored_reviews = np.array(functions.get_reviews_vectors(used_train_negative_reviews, vocabulary))
        positive_vectored_reviews = np.array(functions.get_reviews_vectors(used_train_positive_reviews, vocabulary))

        category_probability_given_category_vectors = naiveBayes.naive_bayes_train(
            [negative_vectored_reviews, positive_vectored_reviews], len(vocabulary) + 1)

        train_vectored_reviews = np.array(functions.get_reviews_vectors(used_train_reviews, vocabulary))
        train_true_positives, train_true_negatives, train_false_positives, train_false_negatives = naiveBayes.naive_bayes_evaluate(train_vectored_reviews, category_probability_given_category_vectors)
        train_accuracy = get_accuracy(train_true_positives, train_true_negatives, train_false_positives, train_false_negatives)
        train_y_values.append(train_accuracy)

        test_vectored_reviews = np.array(functions.get_reviews_vectors(test_reviews, vocabulary))
        test_true_positives, test_true_negatives, test_false_positives, test_false_negatives = naiveBayes.naive_bayes_evaluate(test_vectored_reviews, category_probability_given_category_vectors)
        test_accuracy = get_accuracy(test_true_positives, test_true_negatives, test_false_positives, test_false_negatives)
        test_y_values.append(test_accuracy)

    return x_values, train_y_values, test_y_values


def plot_learning_curve(x_values, train_y_values, test_y_values):
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
