"""
School: University of California, Berkeley
Course: BIOENG 145/245
Author: Yorick Chern
Instructor: Liana Lareau
Assignment 4
SOLUTION
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

    def squared_error(self, X, y):
        """
        Q:  use X, y, and self.theta to calculate the sum-squared-error.

        Inputs
        - X: data matrix (N, D)
        - y: ground truth (N, 1)

        Outputs:
        - loss: loss
        """

        loss = ...
        return loss

    def squared_gradient(self, X_batch, y_batch):
        """
        Q:  calculate the gradient of the squared error w.r.t. self.theta, the weights,
        and self.bias, the bias

        Inputs
        - X_batch: data matrix (batch_size, D)
        - y_batch: ground truth (batch_size, 1)

        Outputs:
        - dldw: gradient w.r.t. weights should be of shape (D, 1)
        - dldb: gradient w.r.t. bias should be of shape (1, 1)
        """
        ...
        assert dldw.shape == (D, 1), f"dldw shape not {D, 1}, but it is {dldw.shape}"
        return dldw, dldb

if __name__ == '__main__':
    N = 200
    D = 1
    X = np.random.rand(N, D) * 20

    # you can change these 3 numbers - this is the line we want to regress
    # y = wx + b + noise
    theta_set = -0.42
    b_set = -0.13
    noise = np.random.randn(N, D) * 0.50

    graph_progress = True      # turn this to True to visualize your code's progress

    y = X * theta_set + b_set + noise   # change the coefficients here


    lin_reg = LinearRegression(gd='mb')
    theta, bias, losses = lin_reg.fit(X, y, lr=5e-5, epochs=1000)   # these parameters work the best

    if graph_progress:
        graph(X, y, title=f"y = {theta_set}x + {b_set} + noise")
        graph_line(X, y, theta, bias)
        graph_losses(losses)

    pass
