"""
School: University of California, Berkeley
Course: BIOENG 145/245
Author: Yorick Chern
Instructor: Liana Lareau
Assignment 4
"""

import numpy as np
import matplotlib.pyplot as plt

def graph(X, y, title=""):
    plt.scatter(X.flatten(), y.flatten(), marker=".")
    plt.title(title)
    plt.show()

def graph_line(X, y, w, b):
    plt.plot(X.flatten(), X @ w + b, color="red")
    plt.scatter(X.flatten(), y.flatten(), marker=".")
    plt.title(f"Linear Regression Results: y = {np.round(w.flatten()[0], 2)}x + {np.round(b.flatten()[0], 2)}")
    plt.show()

def graph_losses(losses):
    epochs = [i for i in range(len(losses))]
    plt.plot(epochs, losses)
    plt.title("Losses vs. Epochs")
    plt.show()


class LinearRegression:

    def __init__(self, gd='mb'):
        """
        The initiatialization function. (Implemented for you).
        
        Inputs
        - gd: can be 'mb' for mini-batch, 'b' for batch, or 'sgd' for stochastic gradient descent
        """
        self.gd = gd
        assert self.gd in ["mb", "b", "sgd"], "gd argument needs to be one of \"mb\", \"b\", or \"sgd\"!"

    def fit(self, X, y, epochs=10, lr=1e-3, batch_size=32):
        """
        Q: this is the main function that will run the entire gradient descent algorithm.

        Inputs
        - X: data matrix (N, D)
        - y: ground truth (N, )
        - epochs: the number of epochs to train for
        - lr: the learning rate
        - batch_size: the size of the batch to use

        Outputs:
        - losses: list of losses over epochs
        """
        N, D = X.shape
        y = y.reshape(N, 1)     # we will reshape y to avoid broadcasting errors

        ...
        return self.theta, self.bias, losses

    def mean_squared_error(self, X, y):
        """
        Q:  use X, y, and self.theta to calculate the mean-squared-error.

        Inputs
        - X: data matrix (N, D)
        - y: ground truth (N, 1)

        Outputs:
        - loss: loss
        """

        loss = ...
        return loss

    def mean_squared_gradient(self, X_batch, y_batch):
        """
        Q:  calculate the gradient of the mean-squared-error w.r.t. self.theta, the weights,
        and self.bias, the bias

        Inputs
        - X_batch: data matrix (batch_size, D)
        - y_batch: ground truth (batch_size, 1)

        Outputs:
        - dldw: gradient w.r.t. weights should be of shape (D, 1)
        - dldb: gradient w.r.t. bias should be of shape (1, 1)
        """
        batch_size, D = X_batch.shape
        dldw = ...
        dldb = ...
        assert dldw.shape == (D, 1), f"dldw shape not {D, 1}, but it is {dldw.shape}"
        return dldw, dldb

if __name__ == '__main__':

    # this section tests each function (besides lin_reg.fit()) individually
    X = np.array([[0.48553452, 0.02705233, 0.59605814, 0.50225229, 0.81243239],
       [0.17469252, 0.25435186, 0.19408888, 0.17743825, 0.13085249],
       [0.28457151, 0.4112227 , 0.37798194, 0.38538792, 0.52813069],
       [0.88580912, 0.67981496, 0.68811359, 0.01019072, 0.24051601],
       [0.43847827, 0.40985491, 0.44461599, 0.38423286, 0.98673558],
       [0.85719302, 0.11396699, 0.88914111, 0.95473146, 0.98221489],
       [0.41547788, 0.22504066, 0.96959725, 0.34318878, 0.76847212],
       [0.749719  , 0.70648943, 0.1026854 , 0.62488796, 0.67939783],
       [0.16971717, 0.89160743, 0.44127767, 0.33990768, 0.28538788],
       [0.42643292, 0.50454711, 0.30491342, 0.19072025, 0.3468241 ]])

    y = np.array([[0.52060233],
       [0.48235778],
       [0.47203745],
       [0.51458632],
       [0.70178445],
       [0.17644808],
       [0.82841289],
       [0.7181598 ],
       [0.80074196],
       [0.50740697]])

    lin_reg = LinearRegression()
    lin_reg.theta = np.array([[0.96348654],
       [0.97798831],
       [0.47966334],
       [0.65893575],
       [0.72210114]])
    lin_reg.bias = np.array([[0.05831978]])

    print(lin_reg.mean_squared_error(X, y))  # solution result: 1.971525196568432
    print(lin_reg.mean_squared_gradient(X, y))

    """
    mean_squared_gradient should look like this
    (array([[1.4944319 ],
       [1.05027943],
       [1.42694079],
       [1.20808437],
       [1.69532283]]), 2.5683096567102917)
    """

    # this section tests the entire class, aka the fit function and produces visualization results
    # N = 200
    # D = 1
    # X = np.random.rand(N, D) * 20

    # # you can change these 3 numbers - this is the line we want to regress
    # # y = wx + b + noise
    # theta_set = -0.42
    # b_set = -0.13
    # noise = np.random.randn(N, D) * 0.50

    # graph_progress = True      # turn this to True to visualize your code's progress

    # y = X * theta_set + b_set + noise   # change the coefficients here


    # lin_reg = LinearRegression(gd='mb')
    # theta, bias, losses = lin_reg.fit(X, y, lr=5e-5, epochs=1000)   # these parameters work the best

    # if graph_progress:
    #     graph(X, y, title=f"y = {theta_set}x + {b_set} + noise")
    #     graph_line(X, y, theta, bias)
    #     graph_losses(losses)

    pass
