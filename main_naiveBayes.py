import functions
import naiveBayes
import metrics_naiveBayes
import numpy as np


def run_naive_bayes():
    while True:
        choice = input("\nNaive Bayes Algorithm\n"
                       "1. Train & Evaluate\n"
                       "2. Plot Learning Curve\n"
                       "3. Test Parameter on Dev Data\n"
                       "0. Exit\n"
                       "Select : ")
        if choice == "1":
            available_words = functions.read_available_words("imdb.vocab")
            reviews, negative_reviews, positive_reviews = functions.read_reviews("train//labeledBow.feat")
            train_negative_reviews, train_positive_reviews, train_reviews, dev_reviews = metrics_naiveBayes.split_train_dev(negative_reviews, positive_reviews)

            word_appearances = functions.get_word_appearances(train_reviews, available_words)
            negative_word_appearances = functions.get_word_appearances(train_negative_reviews, available_words)
            positive_word_appearances = functions.get_word_appearances(train_positive_reviews, available_words)

            vocabulary = functions.get_vocabulary([len(train_negative_reviews), len(train_positive_reviews)], [negative_word_appearances, positive_word_appearances], word_appearances, 1200)

            negative_vectored_reviews = np.array(functions.get_reviews_vectors(train_negative_reviews, vocabulary))
            positive_vectored_reviews = np.array(functions.get_reviews_vectors(train_positive_reviews, vocabulary))

            category_probability_given_category_vectors = naiveBayes.naive_bayes_train([negative_vectored_reviews, positive_vectored_reviews], len(vocabulary) + 1)

            test_reviews, test_negative_reviews, test_positive_reviews = functions.read_reviews("test//labeledBow.feat")

            test_vectored_reviews = np.array(functions.get_reviews_vectors(test_reviews, vocabulary))

            print("\nAlgorithm Evaluation")
            true_positives, true_negatives, false_positives, false_negatives = naiveBayes.naive_bayes_evaluate(test_vectored_reviews, category_probability_given_category_vectors)
            accuracy = metrics_naiveBayes.get_accuracy(true_positives, true_negatives, false_positives, false_negatives)
            print("Algorithm accuracy: {0}\n".format(accuracy))

        elif choice == "2":
            warning = input("\nWarning! This is probably going to take an eternity to finish. Do you wish to proceed? (Y/N)\n"
                            "Select : ")
            if warning == "Y":
                available_words = functions.read_available_words("imdb.vocab")
                reviews, negative_reviews, positive_reviews = functions.read_reviews("train//labeledBow.feat")
                train_negative_reviews, train_positive_reviews, train_reviews, dev_reviews = metrics_naiveBayes.split_train_dev(negative_reviews, positive_reviews)
                test_reviews, test_negative_reviews, test_positive_reviews = functions.read_reviews("test//labeledBow.feat")
                x_values, train_y_values, test_y_values = metrics_naiveBayes.get_learning_curve_points(train_negative_reviews, train_positive_reviews, test_reviews, available_words)
                metrics_naiveBayes.plot_learning_curve(x_values, train_y_values, test_y_values)

        elif choice == "3":
            parameter = input("\nPlease give the parameter with which you want to evaluate the Dev Data\n"
                              "Parameter : ")

            try:
                if isinstance(int(parameter), int):

                    available_words = functions.read_available_words("imdb.vocab")
                    reviews, negative_reviews, positive_reviews = functions.read_reviews("train//labeledBow.feat")
                    train_negative_reviews, train_positive_reviews, train_reviews, dev_reviews = metrics_naiveBayes.split_train_dev(negative_reviews, positive_reviews)

                    word_appearances = functions.get_word_appearances(train_reviews, available_words)
                    negative_word_appearances = functions.get_word_appearances(train_negative_reviews, available_words)
                    positive_word_appearances = functions.get_word_appearances(train_positive_reviews, available_words)

                    vocabulary = functions.get_vocabulary([len(train_negative_reviews), len(train_positive_reviews)], [negative_word_appearances, positive_word_appearances], word_appearances, int(parameter))

                    negative_vectored_reviews = np.array(functions.get_reviews_vectors(train_negative_reviews, vocabulary))
                    positive_vectored_reviews = np.array(functions.get_reviews_vectors(train_positive_reviews, vocabulary))

                    category_probability_given_category_vectors = naiveBayes.naive_bayes_train([negative_vectored_reviews, positive_vectored_reviews], len(vocabulary) + 1)

                    dev_vectored_reviews = np.array(functions.get_reviews_vectors(dev_reviews, vocabulary))

                    print("\nAlgorithm Evaluation")
                    true_positives, true_negatives, false_positives, false_negatives = naiveBayes.naive_bayes_evaluate(dev_vectored_reviews, category_probability_given_category_vectors)
                    accuracy = metrics_naiveBayes.get_accuracy(true_positives, true_negatives, false_positives, false_negatives)
                    print("Algorithm accuracy on Dev Data with parameter {0}: {1}\n".format(parameter, accuracy))

            except ValueError:
                print("\nParameter must be type integer!\n")

        elif choice == "0":
            break

        else:
            print("Invalid Input\n")
