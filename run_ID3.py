from MetricsForID3 import train, get_learning_curve_points, plot_learning_curve, classifier 
from ID3 import ID3
from functions import read_available_words, read_reviews, get_reviews_vectors
import os 
import numpy as np
import time

def run_ID3():

    while True:
        choice = input("\nID3 MENU\n"
              "1 : Print Learning Curves (WARNING: it takes about five minutes!).\n"
              "2 : Run ID3 on train and test data (WARNING: it takes about half a minute!).\n"
              "0 : Exit\n"
              "Select : ")

        if choice == "1":
            """
            :Comments: We read the words from the imdb.vocab, the reviews (general, positive, negative) from the train/labeledBow.feat . 
                       Then, we split the reviews (general, positive, negative) into development reviews and train reviews.
                       Development reviews(general, positive, negative) are the first 5% from the file and train the rest 95% inserted into train reviews
                       (general, positive, negative).
                       The we use the train function to get the vector with the most valuable words (properties), the train data vectored, the negative and 
                       positive length of the reviews that contained, vocabulary with most valuable words, but this time as a dictionary, and finally reviews, 
                       positive reviews and negative reviews.
                       We also make vectors the development reviews and the test reviews (after we read the test reviews).
                       After all these we call the function get_learning_curve_points with train reviews (general, positive, negative), test reviews and 
                       available_words, which function assigns values at the lists(x_values, train_y_values, test_y_values). The lists that i mentioned
                       before are used in the function plot_learning_curve to plot the the learning curves of accuracy and error.
            """
            start_time = time.time()

            available_words = read_available_words("imdb.vocab")

            reviews, negative_reviews, positive_reviews = read_reviews("train//labeledBow.feat")


            pos_percent_05 = round(len(positive_reviews)*0.05)
            neg_percent_05 = round(len(negative_reviews)*0.05)

            train_positive_reviews = positive_reviews[pos_percent_05:]
            train_negative_reviews = negative_reviews[neg_percent_05:]
            train_reviews = train_positive_reviews + train_negative_reviews

            dev_positive_reviews = positive_reviews[:pos_percent_05]
            dev_negative_reviews = negative_reviews[:neg_percent_05]
            dev_reviews = dev_positive_reviews + dev_negative_reviews

            vector, train_data, neg_length, pos_length, vocabulary = \
            train(available_words, train_positive_reviews, train_negative_reviews, train_reviews)

            dev_vectored_reviews = np.array(get_reviews_vectors(dev_reviews, vocabulary))

            test_reviews, test_negative_reviews, test_positive_reviews = read_reviews("test//labeledBow.feat")
            vectored_reviews = np.array(get_reviews_vectors(test_reviews, vocabulary))

            x_values, train_y_values, test_y_values = \
            get_learning_curve_points(train_negative_reviews, train_positive_reviews, train_reviews, test_reviews, available_words)

            plot_learning_curve(x_values, train_y_values, test_y_values)

            print("--- %s seconds ---" % (time.time() - start_time))

        elif choice == "2":
            """
            :Comments: We read the words from the imdb.vocab, the reviews (general, positive, negative) from the train/labeledBow.feat .
                       Then, we split the reviews (general, positive, negative) into development reviews and train reviews.
                       Development reviews(general, positive, negative) are the first 5% from the file and train the rest 95% inserted into train reviews
                       (general, positive, negative)(we dont't set the Development data, but we explain our thinking).
                       Then, we use the train function to get the vector with the most valuable words (properties), the train data vectored, the negative and 
                       positive length of the reviews that contained, vocabulary with most valuable words, but this time as a dictionary, and finally train 
                       reviews, train positive reviews and train negative reviews.
                       After that, we make an object of the ID3 class and we insert the in the tree the vector with the most valuable words, the train data, 
                       positive and negative counters (which counters counts how many positive and negative reviews are in the current node) and default_category.
                       Now in the root variable is stored the root of the tree that constructed with the train data. When we call the classifier function we pass
                       the following arguments: root of the tree, which tree becomes from the train data, vector with the most valuable words and the vectored 
                       reviews. The classifier prints and returns the correct positive, correct negative, false positive and false negatives reviews, as the tree
                       evaluate them.
            """
            start_time = time.time()

            available_words = read_available_words("imdb.vocab")

            reviews, negative_reviews, positive_reviews = read_reviews("train//labeledBow.feat")

            pos_percent_05 = round(len(positive_reviews)*0.05)
            neg_percent_05 = round(len(negative_reviews)*0.05)

            train_positive_reviews = positive_reviews[pos_percent_05:]
            train_negative_reviews = negative_reviews[neg_percent_05:]
            train_reviews = train_positive_reviews + train_negative_reviews

            vector, train_data, neg_length, pos_length, vocabulary = \
            train(available_words, train_positive_reviews, train_negative_reviews, train_reviews)

            m = 'positive'
            tree = ID3()
            root = tree.insert(vector, train_data, neg_length, pos_length, m, 1)
            print("------------------------------Printing for the Train Data------------------------------")
            classifier(root, vector, train_data)
            print("------------------------------Printing for the Test Data-----------------------------")
            test_reviews, test_negative_reviews, test_positive_reviews = read_reviews("test//labeledBow.feat")
            vectored_reviews = np.array(get_reviews_vectors(test_reviews, vocabulary))
            classifier(root, vector, vectored_reviews)
            print("--- %s seconds ---" % (time.time() - start_time))

        elif choice == '0':
            break

        else:
            print("WRONG ANSWER!\n")
            break