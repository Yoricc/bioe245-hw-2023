"""
School: University of California, Berkeley
Course: BIOENG 145/245
Author: Yorick Chern
Instructor: Liana Lareau
Assignment 3
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product

def read_data(file_name, test_size=0.2):
    """
    Q:  read the data from the q3_data.csv.
        the data contains 2 columns with 1 column being the sequence and the second column being
        either 1 or 0. Then, shuffle the data and create a training and testing set based on test_size.

        To read the data, I recommend using pandas' pd.read_csv() function.
        To shuffle and split the data into training/testing sets, I recommend sklearn's train_test_split

        You can use any methods you want to read in the dataset as long as you produce the correct
        output.

        Note, in the case that we are reading q3_data.csv, there should be 5000 sequences.

    Inputs
    - file_name: a string, the name of the data file ("q3_data.csv")
    - test_size: a float to show how many

    Outputs
    - X_train: np.array with shape (N * (1 - test_size)); each item is a sequence from the dataset
    - X_test: np.array with shape (N * test_size)
    - y_train: np.array with shape (N * (1 - test_size)); y[i] is X[i]'s ground-truth label
    - y_test: np.array with shape (N * test_size)
    """

    # TODO: use your favorite methods to complete this section - no restriction (do not import any foreign packages, but train_test_split is ok)

    assert type(X_train) == type(X_test) == type(y_train) == type(y_test) == np.ndarray, f"read_data() NEEDS to output np.ndarray, instead it's {type(X_train)}!"
    return X_train, X_test, y_train, y_test

def build_transition_matrix(data, k):
    """
    Q:  Here, we will build a transition matrix for a set of sequences.
        Suppose we have a sequence "ACTAGCTACT..." and k = 3, then the 
        list of states for this sequence will be:
        ex: states = [ACT, CTA, TAC, ACT,...]. 
        
        In order to make it easier for ourselves, we will create a dictionary
        mapping every kmer state to an integer label. So, if our
        kmer2idx dictionary is {'ACT' : 0, 'CTA' : 1, 'TAC' : 2, ...}
        then the states above can be written as [0, 1, 2, 1 ..]
        
        Next, we build a 2D transition matrix where
        trans_prob[i, j] = probability of state i transitioning to state j.
        ex: trans_prob[3, 1] = probability of state 3 to state 1. 

        How do we get this number? Say we want to find trans_prob[1, 2], which is the transition probability
        of 'CTA' to 'TAC'. We count how many transitions from 'CTA' to 'TAC' there are and divide
        this number by the TOTAL number of transitions in the entire dataset, and this gives us
        Pr['CTA' | 'TAC']. However, notice that 'CTA' to 'TAC' in reality is just 'CTAC', we can also consider rewrite this
        as Pr['C' | 'CTA']. Then, we define our transition matrix as a (64, 4) matrix where each row is a possible 3-mer
        and each of the 4 columns is A, G, T, or C. Then, we realize that:
        Pr['A' | 'CTA'] + Pr['T' | 'CTA'] + Pr['G' | 'CTA'] + Pr['C' | 'CTA'] = 1

    Inputs
    - data: np.array with shape (training size); each item is a sequence from the dataset
    - kmer: an int describing the kmer we are interested in

    Outputs
    - kmer2idx: a dictionary that gives us the index of each kmer (ex: kmer2idx['ATC'] = 2 and
                kmer2idx['TCG'] = 4, then trans_mat[2][4] is the probability of "ATCG" happening, 
                given that we started with "ATC")
                this is important as it helps us understand what the transition matrix (trans_probs)
                means.
    - trans_probs: np.array with shape (# of states, # of states) with the properties described above
    """


    # Initialize transition probability matrix & generate all possible kmers
    trans_prob = ...
    kmers = ...     # generate a list of all possible kmers combinations, use itertools.product() - read the docs
    # build a dictionary to map each kmer to an integer, this will allow us to keep track of where each kmer
    # is located in the transition matrix (trans_prob)
    kmer2idx = {}   # kmer2idx['ATC'] = 4, for example
    nt2idx = {'A': 0, 'G': 1, 'T': 2, 'C': 3}   # do not modify this
    
    # Iterate through the data to count each transition
    for seq in data:
        ...

    trans_prob = ... # apply 1 pseudocount

    # Normalize transition matrix
    trans_prob =  ...

    return kmer2idx, trans_prob


def log_odds_ratio(seq, k, pos_kmer2idx, pos_trans_probs, neg_kmer2idx, neg_trans_probs):
    """
    Q:  this function will calculate the log odds ratio of a sequence with the following formula

    log_odds = log(probability of being class 1 / probability of being class 0)

    Inputs
    - seq: a string, the sequence to be classified
    - k: an int, the kmer substring length
    - pos_kmer2idx: the index system dictionary for the positive (class 1) transition matrix
    - pos_trans_probs: the transition matrix for the positive sequences (class 1 sequences)
    - neg_kmer2idx: same logic as pos_kmer2idx but for the negative sequences (class 0 sequences)
    - neg_trans_probs: same logic as neg_trans_probs but for the negative sequences (class 0 sequences)

    Outputs
    - score: a float, the log odds ratio
    """
    nt2idx = {'A': 0, 'G': 1, 'T': 2, 'C': 3}
    score = 0
    for i in range(...):
        # calculate the log odds ratio using the variables passed
    return score


def classify(seq, k, pos_kmer2idx, pos_trans_probs, neg_kmer2idx, neg_trans_probs):
    """
    Q:  takes a sequence and classifies whether the sequence is a positive class or a negative class.
        if log_odds_ratio > 0, (probability of positive > negative) ==> classify as positive class!
        else if log_odds_ratio < 0 ==> classify as negative class!

    Inputs:
    (check the function above, they have the same specs)

    Outputs:
    - returns an integer, 0 or 1
    """

    # TODO

    pass

# DO NOT MODIFY THIS
def main(k):
    """
    To make your things smoother, I included this function to help you debug and test your code out.
    Unless you are 100% sure with what you're doing, try not to alter any code from this section.
    This code should split the dataset, train the Markov Chain on the training set, test the MC
    on the testing set, and generate predictions of q3_validation_data.csv that you will upload to
    Gradescope for autograding. If your test set achieves an accuracy of 0.97+, you should be good
    to submit your predictions to the autograder.

    Note: You do not have the ground truth labels for q3_validation_data.csv.
    """
    print("\nRunning Markov Chain with k = {}".format(k))
    # split the dataset
    X_train, X_test, y_train, y_test = read_data("q3_data.csv")

    # find positive classes (class 1) and build the transition matrix
    print("Building positive transition matrix...")
    positive_idx = np.where(y_train == 1)[0]
    pos_kmer2idx, pos_trans_mat = build_transition_matrix(X_train[positive_idx], k)
    assert np.abs(np.mean(np.sum(pos_trans_mat, axis=1)) - 1.0) < 1e-3, f"Positive transition matrix needs to have every row sum to 1! Instead it's {np.mean(np.sum(pos_trans_mat, axis=1))}"

    # find negative classes (class 0) and build the transition matrix
    print("Building negative transition matrix...")
    negative_idx = np.where(y_train == 0)[0]
    neg_kmer2idx, neg_trans_mat = build_transition_matrix(X_train[negative_idx], k)
    assert np.abs(np.mean(np.sum(neg_trans_mat, axis=1)) - 1.0) < 1e-5, f"Negative transition matrix needs to have every row sum to 1! Instead it's {np.mean(np.sum(neg_trans_mat, axis=1))}"

    print("Classifying testing dataset...")
    y_pred = np.zeros_like(y_test)
    for i in range(X_test.shape[0]):
        seq = X_test[i]
        y_pred[i] = classify(seq, k, pos_kmer2idx, pos_trans_mat, neg_kmer2idx, neg_trans_mat)

    accuracy = accuracy_score(y_test, y_pred)
    print("Testing Accuracy: {}".format(accuracy))
    acc = [None, 0.85, 0.98, 0.98, 0.96, 0.93]
    if accuracy < acc[k]:
        print("(note: you need an accuracy of at least {} for k = {} in order to pass the autograder!".format(acc[k], k))

    # classify hidden dataset for autograder
    # loading predictions to q3_predictions.npy
    print("Fetching autograder validation data...")
    test_data = pd.read_csv("q3_validation_data.csv", sep=' ', header=None)
    test_data = test_data.to_numpy().squeeze()

    print("Classifying autograder data...")
    pred = np.zeros(test_data.shape[0])
    for i in range(test_data.shape[0]):
        seq = test_data[i]
        pred[i] = classify(seq, k, pos_kmer2idx, pos_trans_mat, neg_kmer2idx, neg_trans_mat)
    
    print("Loading predictions to q3_predictions.npy...\n")
    np.save("q3_predictions_k={}".format(k), pred)

    """----- Instructor Version ----- (DO NOT INCLUDE IN THE HW)"""
    # test_label = np.load("q3_val_labels.npy")
    # accuracy = accuracy_score(test_label, pred)
    # print("Autograder Accuracy: {}".format(accuracy))

if __name__ == '__main__':
    for k in range(1, 6):
        main(k)
    
    print("Markov Chain Assignment Complete!")

    print("\n===== SUBMISSION INSTRUCTIONS! =====")
    print("Don't forget to submit all the q3_predictions_k=*.npy along with your q1.py, q2.py, and q3.py!")
    print("WEHN SUBMITTING TO GRADESCOPE, PLEASE DO NOT INCLUDE THE .csv files!!!")
    print("DO NOT CHANGE ANY FILE NAMES, OR ELSE THE AUTOGRADER WILL NOT BE ABLE TO FIND YOUR FILE!!")
