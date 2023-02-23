"""
School: University of California, Berkeley
Course: BIOENG 145/245
Author: Yorick Chern
Instructor: Liana Lareau
Assignment 4
"""

import numpy as np

"""Q 1.1"""
def accuracy_score(y_pred, y_true):
    """
    Q:  given the predictions (y_pred), calculate the accuracy score comparing
        to the actual answer (y_true).
        ex: accuracy_score([1, 0, 1, 1, 0], [1, 0, 1, 0, 0]) = 0.80
        in layman terms, a function that performs what you do when you grade a multiple choice exam!
    
    Inputs
    - y_pred: list of predictions => np.array with shape (N, )
    - y_true: list of ground truth labels => np.array with shape (N, )

    Outputs
    - accuracy: a float (correct prediction divided by total number of samples N)
    """

    assert len(y_pred.shape) == 1 and y_pred.shape == y_true.shape, "dimension mismatch"
    accuracy = ...
    return accuracy

"""Q 1.2"""
def train_test_split(X, y, test_size):
    """
    Q:  create a function that shuffles the entire dataset, breaks it into training/test sets
        accordingly to the test_size given.
        ex: X.shape = (100, 5) => (100 sample points, 5 features each)
            y.shape = (100, )
        X.train = (80, 5)
        y.train = (80, )
        X.test = (20, 5)
        y.test = (20, )

    Inputs
    - X: np.array (N, D) data
    - y: np.array (N, ) ground truth label => X[i]'s label is y[i]
    - test_size: a float between 0 and 1. test_size=0.2 means we are performing a 80-20% train-test split

    Outputs
    - X_train: np.array with shape (N * (1 - test_size), D)
    - X_test: np.array with shape (N * test_size, D)
    - y_train: np.array with shape (N * (1 - test_size))
    - y_test: np.array with shape (N * test_size)
    """

    ...
    return X_train, X_test, y_train, y_test

"""Q 1.3"""
def k_fold_cv(X, y, k_folds=5, shuffle=True):
    """
    Q:  shuffle (if shuffle=True) the dataset and divide X & y into k folds and do the following procedure.
        ex: partition X into X1, X2, ..., Xk
        then, return a dataset of k items, where each Xi is the test set and 
        the rest is the training set, like so:

        dataset[1] = train([X2, X3, .., Xk]) + test(X1)
        dataset[2] = train([X1, X3, .., Xk]) + test(X2)
        dataset[3] = train([X1, X2, X4, .., Xk]) + test(X3)
        dataset[i] = train([X1, .., X_(i-1), X_(i+1), .., Xk]) + test(Xi)
        ...
        dataset[k] = train([X1, X3, .., X_(k-1)]) + test(Xk)

        example output:
        >>> X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> dataset = k_fold_cv(X, y, k_fold=5, shuffle=False)
        >>> dataset
            [        
                    X_train:  [2. 3. 4. 5. 6. 7. 8. 9.]
                    X_test:  [0. 1.]
                    y_train:  [2. 3. 4. 5. 6. 7. 8. 9.]
                    y_test:  [0. 1.],

                    X_train:  [0. 1. 4. 5. 6. 7. 8. 9.]
                    X_test:  [2. 3.]
                    y_train:  [0. 1. 4. 5. 6. 7. 8. 9.]
                    y_test:  [2. 3.],

                    X_train:  [0. 1. 2. 3. 6. 7. 8. 9.]
                    X_test:  [4. 5.]
                    y_train:  [0. 1. 2. 3. 6. 7. 8. 9.]
                    y_test:  [4. 5.],

                    X_train:  [0. 1. 2. 3. 4. 5. 8. 9.]
                    X_test:  [6. 7.]
                    y_train:  [0. 1. 2. 3. 4. 5. 8. 9.]
                    y_test:  [6. 7.],

                    X_train:  [0. 1. 2. 3. 4. 5. 6. 7.]
                    X_test:  [8. 9.]
                    y_train:  [0. 1. 2. 3. 4. 5. 6. 7.]
                    y_test:  [8. 9.]
            ]

            Visual aid: https://www.researchgate.net/profile/Adel-Elsharkawy/publication/360195188/figure/fig5/AS:1169356675399692@1655807765667/Comparison-between-the-holdout-method-and-the-5K-fold-cross-validation-method.ppm
    
    Inputs
    - X: np.array (N, D) data
    - y: np.array (N, ) ground truth label => X[i]'s label is y[i]
    - k_folds: how many folds we want to divide our dataset into

    Outputs
    - dataset: a list with k items where each item is a tuple
    """

    ...
    return dataset

"""Q 1.4"""
def normalize(X):
    """
    Q:  normalize the dataset X along the feature dimension.
        use the formula: (X - mean) / standard_deviation

    Inputs
    - X: np.ndarray with shape (N, D)
    
    Outputs
    - X_norm: normalized X, an np.ndarray with shape (N, D)
    """
    ...
    return X_norm

if __name__ == '__main__':
    # print(accuracy_score(np.array([1, 0, 1, 1, 0]), np.array([1, 0, 1, 0, 0])))


    X = np.linspace(0, 9, 10).reshape(10, 1)
    # X = np.concatenate([X, X, X], axis=1)     # uncomment this to test out multiple dimensions of X
    y = np.linspace(0, 9, 10)


    # uncomment this section to test out train_test_split()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)
    # print(X_train, y_train, np.all(X_train[:, 0].flatten() == y_train))
    # print(X_test, y_test, np.all(X_test[:, 0].flatten() == y_test))


    # uncomment this section to test out k_fold_cv()
    # dataset = k_fold_cv(X, y, k_folds=5)
    # for data in dataset:
    #     X_train, X_test, y_train, y_test = data
    #     print("X_train: ", X_train.flatten())
    #     print("X_test: ", X_test.flatten())
    #     print("y_train: ", y_train)
    #     print("y_test: ", y_test, ",\n")