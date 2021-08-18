import functions
import adaboost
import numpy as np
import time
from metrics_adaboost import plot_learning_curve


def run_adaboost():

    while True:

        choice = input("\n~AdaBoost Menu~\n"
                       "1 : Train and evaluate in default test data\n"
                       "2 : Print learning curves\n"
                       "0 : Exit \n"
                       "Select : ")

        if choice == '1' :

            print("\nLoading Adaboost...")
            print("Estimated training time : 12'")
            start_time = time.time()

            # Read Available Words
            available_words = functions.read_available_words("imdb.vocab")

            # Get Vocabulary
            train_reviews, negative_reviews, positive_reviews = functions.read_reviews("train//labeledBow.feat")

            word_appearances = functions.get_word_appearances(train_reviews, available_words)
            negative_word_appearances = functions.get_word_appearances(negative_reviews, available_words)
            positive_word_appearances = functions.get_word_appearances(positive_reviews, available_words)

            vocabulary = functions.get_vocabulary([len(negative_reviews), len(positive_reviews)],
                                                  [negative_word_appearances, positive_word_appearances], word_appearances,
                                                  300)

            # Get np array of reviews
            train_reviews_vector = np.array(functions.get_reviews_vectors(train_reviews, vocabulary))

            # Split data to train and development
            train_reviews_vector, development_reviews_vector = split_data(train_reviews_vector)

            # Use x percentage of train data
            percentage = 1
            n_reviews = train_reviews_vector.shape[0]
            break_point = int(n_reviews * percentage)
            train_reviews_vector = np.vstack((train_reviews_vector[0:int(break_point / 2)],
                                              train_reviews_vector[
                                              int(n_reviews / 2):int(n_reviews / 2 + break_point / 2)]))

            train_values = train_reviews_vector[:, 0]
            development_values = development_reviews_vector[:, 0]

            # Get test reviews
            test_reviews = functions.read_reviews("test//labeledBow.feat", flag=True)
            test_reviews_vector = np.array(functions.get_reviews_vectors(test_reviews, vocabulary))

            test_values = test_reviews_vector[:, 0]

            # Get classifiers by training 95% of train data
            classifiers = adaboost.train(train_reviews_vector[:, 1:], train_values, len(vocabulary), 600)

            # Train predictions
            train_predictions = adaboost.predict(train_reviews_vector[:, 1:], classifiers)
            correctly_classified = sum(train_predictions == train_values)
            perc = correctly_classified / train_values.shape[0] * 100
            print("%f%% of train reviews were correctly classified" % perc)

            # Test predictions
            predictions = adaboost.predict(test_reviews_vector[:, 1:], classifiers)
            correctly_classified = sum(predictions == test_values)
            perc = correctly_classified / test_values.shape[0] * 100
            print("%f%% of test reviews were correctly classified" % perc)

            # Print runtime
            print("Total runtime : %s sec." % (time.time() - start_time))

        elif choice == '2' :
            plot_learning_curve()

        elif choice == '0' :
            break

        else:
            print("\nInvalid input\n")
            break



def split_data(vector):
    n_reviews = vector.shape[0]
    break_point = int(n_reviews * 0.95)

    train_data = np.vstack((vector[0:int(break_point/2)], vector[int(n_reviews/2):int(n_reviews/2 + break_point/2)]))
    development_data = np.vstack((vector[int(break_point/2):int(n_reviews/2)], vector[int(n_reviews/2+break_point/2):n_reviews]))
    return train_data, development_data