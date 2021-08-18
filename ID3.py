import numpy as np

class Node:
    def __init__(self, vector, train_data, pos, neg, default_category):
        """
        :param vector: Is the vector with the properties.
        :param train_data: A 2d numpy array with shape ( len(reviews), len(vocabulary)+1 ) where each row represents one 
                           review. Every review is a binary numpy array. The first item indicates the category of the 
                           review and the following items whether or not this review contains a specific index (word).
        :param pos: A counter that contains the size of the positive reviews
        :param neg: A counter that contains the size of the negative reviews
        :param default_category: A string that is either "positive" either "negative"(In the first call from the main, 
                                 default_category is "positive")
        """
        self.left = None
        self.right = None
        self.vector = vector
        self.train_data = train_data
        self.pos = pos
        self.neg = neg
        self.default_category = default_category
    
class ID3:
    def insert(self, vector, train_data, pos, neg, default_category, i):
        """
        :param vector: Is the vector with the properties.
        :param train_data: A 2d numpy array with shape ( len(reviews), len(vocabulary)+1 ) where each row represents one 
                           review. Every review is a binary numpy array. The first item indicates the category of the 
                           review and the following items whether or not this review contains a specific index (word).
        :param pos: A counter that contains the size of the positive reviews
        :param neg: A counter that contains the size of the negative reviews
        :param default_category: A string that is either "positive" either "negative"(In the first call from the main, 
                                 default_category is "positive")
        :param i: "i" is an extra parameter that is used to check if each one of the reviews (each_train) that I examine now 
                  contains the property (which is stored in the vector). len(review) = len(vector) + 1, review[0] = 1=>positive 
                  reviews[0] = 0=>negative 
                  (In the first call from the main, i = 1). 
                  So, in each reconstruction, I increase the variable i to check the next property.
                  I only check if review[i] == 1 and I am not using the vector, because the train_data length depends on vector
                  and also train_data constructed after the vector's construction.
                  E.g. If vector contains 100 words(probably the indexes of 100 words), then each_train in train_data will be 
                  a vectored_review that has 101 values. First value to separate if train is positive or negative, second value 
                  to separate if the word with the biggest IG(which is the first in vector) is in the review(each_train),
                  third etc. That's also the reason that i am not taking vector[0] to see if the word is in the review
                  (each_train). Because I already know by accessing each review if the review(each_train) contains the word that
                  described in the vector. I just use a vector as a counter whose length is reduced by one in each division.
        """
        node = Node(vector, train_data, pos, neg, default_category)
        
        if len(train_data) == 0:
            return node
        elif pos > 0.95 * len(train_data):
            return node
        elif neg > 0.95 * len(train_data):
            return node
        elif len(vector) == 0:
            if pos > neg:
                return node
            elif neg > pos:
                return node
        else:
            vector = np.delete(vector, 0)
            left_data = []
            right_data = []
            left_counter_neg = 0
            left_counter_pos = 0
            right_counter_neg = 0
            right_counter_pos = 0
            for each_train in train_data:
                if each_train[i] == 1:
                    left_data.append(each_train)
                    if each_train[0] == 1:
                        left_counter_pos += 1
                    else:
                        left_counter_neg += 1
                else:
                    right_data.append(each_train)
                    if each_train[0] == 1:
                        right_counter_pos += 1
                    else:
                        right_counter_neg += 1
            if left_counter_pos > left_counter_neg:
                m1 = "positive"
            else:
                m1 = "negative"
            if right_counter_pos > right_counter_neg:
                m2 = "positive"
            else:
                m2 = "negative"
            left_data = np.array(left_data)
            right_data = np.array(right_data)
            l = i + 1
            node.left = self.insert(vector, left_data, left_counter_pos, left_counter_neg, m1, l)
            r = i + 1
            node.right = self.insert(vector, right_data, right_counter_pos, right_counter_neg, m2, r)
            return node
