import numpy as np


class DecisionStump:

    def __init__(self):
        self.word_index = None
        self.positive_value = 1
        self.a = None

    def make_hypothesis(self, reviews):

        """
        :param reviews: Sample reviews
        :return: hypothesis about the real value of the review
        """

        N = reviews.shape[0]
        best_column = reviews[:, self.word_index]
        hypothesis = np.ones(N)
        if self.positive_value == 1:
            hypothesis[best_column == 0] = 0
        else:
            hypothesis[best_column == 1] = 0

        return hypothesis

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


def train(reviews, values, n_words, n_hypothesis) :

    """
    :param reviews: Sample reviews
    :param values: Reviews' values
    :param n_words: Number of words in vocabulary
    :param n_hypothesis: Number of classifiers made
    :return: Array of classifiers
    """

    # Initialise number of examples, number of words
    N = reviews.shape[0]
    n_words = n_words
    M = n_hypothesis

    # Initialise weights and classifiers
    weights = np.full(N, 1 / N)
    classifiers = np.empty(M, dtype=DecisionStump)

    print("Total number of classifiers : %d"%M)

    for _ in range(M):

        print("HYPOTHESIS ", _)

        stump = DecisionStump()

        # Initialise list with word errors
        words = []

        for word_column in range(n_words):
            word_vector = reviews[:, word_column]
            positive_value = 1
            hypothesis = np.ones(N)
            hypothesis[word_vector != positive_value] = 0

            # Calculate error
            error = sum(weights[hypothesis != values])

            # Switch positive value to 0 if error > 0.5
            if error > 0.5:
                positive_value = 0
                error = 1 - error

            words.append((error, positive_value))

        errors = list(map(lambda x: x[0], words))
        stump.word_index = errors.index(min(errors))
        t_error, stump.positive_value = words[stump.word_index]

        # Calculate performance of stump m in final classification
        stump.a = 0.5 * np.log((1 - t_error) / t_error)

        # Make best_word_hypothesis
        hypothesis = stump.make_hypothesis(reviews)

        # Calculate new weights and normalize them
        weights = np.array([weights[i] * np.exp(stump.a) if hypothesis[i] != values[i]
                            else 0.982 * weights[i] * np.exp(-1 * stump.a) for i in range(N)])
        weights /= np.sum(weights)

        classifiers[_] = stump

    return classifiers


def predict(reviews, classifiers):

    """
    :param reviews: Sample reviews
    :param classifiers: Classifiers extracted by train data
    :return: Final prediction of all classifiers for the values of reviews
    """

    a_sum = 0
    for stump in classifiers:
        a_sum += stump.a
    threshold = a_sum / 2

    predictions = [stump.a * stump.make_hypothesis(reviews) for stump in classifiers]
    predictions_sums = np.sum(predictions, axis = 0)
    final_prediction = predictions_sums > threshold

    return final_prediction

