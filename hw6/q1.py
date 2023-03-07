"""
School: University of California, Berkeley
Course: BIOENG 145/245
Author: Yorick Chern
Instructor: Liana Lareau
Assignment 6
"""

import numpy as np

def conv_forward(X, w, b, stride=1):
    """
    Q:  implement the linear forward pass of a convolutional layer (assume no padding)
        H & W = height and width of the original image (data matrix)
        HH & WW = height and width of the kernel filter
        C = number of channels (ex: an RGB image has 3 channels)
        C' = number of kernels

    Inputs
    - X: data matrix with the shape (N, C, H, W) => (batch_size, channels, height, width)
    - w: kernel matrix with the shape (C', C, HH, WW); C' is the number of kernels while C is the old
         number of channels
    - b: one bias per kernel, so (C')

    Outputs
    - Z: output after running the convolutional layer should be of shape (N, C', H_out, W_out)
    """

    assert stride >= 1 and type(stride) == int, f"strides need to be an integer and >= 1, but it's {stride}"
    N, C, H, W = X.shape
    C_out, _, HH, WW = w.shape
    ...      
    return out



if __name__ == '__main__':

    def close_enough(val1, val2, thresh=1e-12):
        return np.all(np.abs(val1 - val2) < thresh)

    X = np.load("X.npy")
    w = np.load("w.npy")
    b = np.load("b.npy")


    out = conv_forward(X, w, b, 1)
    test_out = np.load("q1_stride1.npy")
    assert close_enough(out, test_out), "stride = 1 incorrect output!"
    print("Stride =  1 passed...")

    out = conv_forward(X, w, b, 3)
    test_out = np.load("q1_stride3.npy")
    assert close_enough(out, test_out), "stride = 3 incorrect output!"
    print("Stride =  3 passed...")

    out = conv_forward(X, w, b, 13)
    test_out = np.load("q1_stride13.npy")
    assert close_enough(out, test_out), "stride = 13 incorrect output!"
    print("Stride = 13 passed...")

    print("All test cases passed!")
    pass
    

