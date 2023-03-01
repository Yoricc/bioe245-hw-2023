"""
School: University of California, Berkeley
Course: BIOENG 145/245
Author: Yorick Chern
Instructor: Liana Lareau
Assignment 5
"""
import numpy as np

def linear_forward(X, w, b):
    """
    Q:  calculate the output of a node with no activation function: Z = Xw + b 

    Inputs
    - X: input matrix (N, D)
    - w: linear node weight matrix (D, D')
    - b: a scalar bias term
    """

    ...
    return ...

def linear_backward(X, w, b):
    """
    Q:  with Z = Xw + b, find dZ/dw and dZ/db

    Outputs a tuple of...
    - (dZdw, dZdb)
    """
   	...
    return ...

def relu_forward(X):
    """
    Q:  Z = relu(X) = max(X, 0)

    Inputs
    - X: input matrix (N, D)
    """
    ...
    return ...

def relu_backward(X):
    """
    Q:  Z = relu(X) = max(X, 0), find dZ/dX
    """
    ...
    return ...

def softmax_forward(X):
    """
    Q:  Z = softmax(X)

    Inputs
    - X: input matrix (N, D)
    """
    N, D = X.shape
    ...
    return ...

def softmax_backward(X):
    """
    Q:  Z = softmax(X), find dZ/dX
    """
    N, D = X.shape
    ...
    return ...

if __name__ == '__main__':

    # due to popular demand, we have increased the number and clarity of test cases available to you

    # we will use this (5, 8) matrix to test all the functions above!
    X = np.array([[-0.38314889,  0.35891731,  0.09037955,  0.98397352, -0.74292248, -0.5883056,  0.54354937,  0.79001348],
                  [ 0.58758113, -0.412598  ,  0.08740239, -0.68723605, -0.29251551, 0.36521658, -0.25330565,  0.03919754],
                  [ 0.97960327, -0.41368028,  0.26308195,  0.94303171, -0.92383705, 0.28187289,  0.35914219, -0.46526478],
                  [ 0.2583081 ,  0.97956892,  0.31049517, -0.68557195, -0.68612885, -0.9054485, -0.70507179,  0.11431403],
                  [-0.7674351 ,  0.69421738, -0.8007104 ,  0.93470719,  0.61132148, 0.54328029,  0.00919623, -0.34544161]])

    # predefined weights and biases to ease your testing
    w = np.array([[0.77805922, 0.67805674, 0.18799035, 0.93644034, 0.87466635, 0.66450703],
                  [0.86038224, 0.21901606, 0.87774923, 0.21039304, 0.76061141, 0.37033866],
                  [0.49032109, 0.71247207, 0.61826719, 0.37348737, 0.4197679 , 0.70488014],
                  [0.37720786, 0.39471295, 0.68555261, 0.48458372, 0.29309447, 0.01436672],
                  [0.68969515, 0.10709357, 0.02608303, 0.35893371, 0.53729841, 0.53873035],
                  [0.31109099, 0.99274133, 0.78935902, 0.77859174, 0.02639908, 0.17466261],
                  [0.30502676, 0.07085277, 0.03068556, 0.4183926 , 0.07385148, 0.99708494],
                  [0.87156768, 0.47651573, 0.76058837, 0.1566234 , 0.95023629, 0.78754312]])
    b = np.array([-0.41458132])

    N, D = X.shape

    # forward pass
    linear_test = linear_forward(X, w, b)
    print("linear_test")
    print(linear_test)

    """
    linear_test:
    array([[ 0.17053049, -0.39162103,  0.69266628, -0.5608487 ,  0.22576308, 0.20272242],
           [-0.66000863,  0.01644713, -0.78068007, -0.17201513, -0.5081938 , -0.44069044],
           [-0.36904111,  0.70333812,  0.07116782,  0.93621757, -0.3900978 , -0.17462756],
           [-0.34748938, -1.04211743, -0.45154255, -1.41118976,  0.14981774, -0.81192474],
           [-0.16206085, -0.54359526,  0.37856668, -0.24113051, -0.60472417, -1.05708054]])
    """

    relu_test = relu_forward(linear_test)
    print("relu_test")
    print(relu_test)

    """
    relu_test:
    array([[0.17053049, 0.        , 0.69266628, 0.        , 0.22576308, 0.20272242],
           [0.        , 0.01644713, 0.        , 0.        , 0.        , 0.        ],
           [0.        , 0.70333812, 0.07116782, 0.93621757, 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.14981774, 0.        ],
           [0.        , 0.        , 0.37856668, 0.        , 0.        , 0.        ]])
    """

    softmax_test = softmax_forward(relu_test)
    print("softmax_test")
    print(softmax_test)

    """
    softmax:
    array([[0.15476137, 0.13049748, 0.26086947, 0.13049748, 0.16354971, 0.1598245 ],
           [0.16620729, 0.16896353, 0.16620729, 0.16620729, 0.16620729, 0.16620729],
           [0.11567963, 0.23372908, 0.12421232, 0.2950197 , 0.11567963, 0.11567963],
           [0.16229491, 0.16229491, 0.16229491, 0.16229491, 0.18852543, 0.16229491],
           [0.1547942 , 0.1547942 , 0.22602898, 0.1547942 , 0.1547942 , 0.1547942 ]])
    """

    assert np.all(np.abs(np.sum(softmax_test, axis=1) - 1.0) < 1e-10), "Rows of softmax output need to sum to 1!"

    # backward pass
    softmax_back_test = softmax_backward(relu_test)
    print(softmax_back_test)

    """
    softmax_back_test:
    """
    sm = np.array([[[ 0.13081029, -0.02019597, -0.04037252, -0.02019597, -0.02531118, -0.02473466],
                   [-0.02019597,  0.11346789, -0.03404281, -0.01702959, -0.02134282, -0.02085669],
                   [-0.04037252, -0.03404281,  0.19281659, -0.03404281, -0.04266513, -0.04169333],
                   [-0.02019597, -0.01702959, -0.03404281,  0.11346789, -0.02134282, -0.02085669],
                   [-0.02531118, -0.02134282, -0.04266513, -0.02134282, 0.1368012 , -0.02613925],
                   [-0.02473466, -0.02085669, -0.04169333, -0.02085669, -0.02613925,  0.13428063]],

                  [[ 0.13858243, -0.02808297, -0.02762486, -0.02762486, -0.02762486, -0.02762486],
                   [-0.02808297,  0.14041486, -0.02808297, -0.02808297, -0.02808297, -0.02808297],
                   [-0.02762486, -0.02808297,  0.13858243, -0.02762486, -0.02762486, -0.02762486],
                   [-0.02762486, -0.02808297, -0.02762486,  0.13858243, -0.02762486, -0.02762486],
                   [-0.02762486, -0.02808297, -0.02762486, -0.02762486, 0.13858243, -0.02762486],
                   [-0.02762486, -0.02808297, -0.02762486, -0.02762486, -0.02762486,  0.13858243]],

                  [[ 0.10229785, -0.02703769, -0.01436884, -0.03412777, -0.01338178, -0.01338178],
                   [-0.02703769,  0.1790998 , -0.02903203, -0.06895468, -0.02703769, -0.02703769],
                   [-0.01436884, -0.02903203,  0.10878362, -0.03664508, -0.01436884, -0.01436884],
                   [-0.03412777, -0.06895468, -0.03664508,  0.20798308, -0.03412777, -0.03412777],
                   [-0.01338178, -0.02703769, -0.01436884, -0.03412777, 0.10229785, -0.01338178],
                   [-0.01338178, -0.02703769, -0.01436884, -0.03412777, -0.01338178,  0.10229785]],

                  [[ 0.13595528, -0.02633964, -0.02633964, -0.02633964, -0.03059672, -0.02633964],
                   [-0.02633964,  0.13595528, -0.02633964, -0.02633964, -0.03059672, -0.02633964],
                   [-0.02633964, -0.02633964,  0.13595528, -0.02633964, -0.03059672, -0.02633964],
                   [-0.02633964, -0.02633964, -0.02633964,  0.13595528, -0.03059672, -0.02633964],
                   [-0.03059672, -0.03059672, -0.03059672, -0.03059672, 0.15298359, -0.03059672],
                   [-0.02633964, -0.02633964, -0.02633964, -0.02633964, -0.03059672,  0.13595528]],

                  [[ 0.13083296, -0.02396125, -0.03498798, -0.02396125, -0.02396125, -0.02396125],
                   [-0.02396125,  0.13083296, -0.03498798, -0.02396125, -0.02396125, -0.02396125],
                   [-0.03498798, -0.03498798,  0.17493988, -0.03498798, -0.03498798, -0.03498798],
                   [-0.02396125, -0.02396125, -0.03498798,  0.13083296, -0.02396125, -0.02396125],
                   [-0.02396125, -0.02396125, -0.03498798, -0.02396125, 0.13083296, -0.02396125],
                   [-0.02396125, -0.02396125, -0.03498798, -0.02396125, -0.02396125,  0.13083296]]])

    assert np.all(np.abs(sm - softmax_back_test) < 1e-8), "Softmax gradient incorrect!"

    relu_back_test = relu_backward(relu_test)
    print("relu_back_test")
    print(relu_back_test)
    """
    relu_back_test:
    """
    rbt = np.array([[1., 0., 1., 0., 1., 1.],
                    [0., 1., 0., 0., 0., 0.],
                    [0., 1., 1., 1., 0., 0.],
                    [0., 0., 0., 0., 1., 0.],
                    [0., 0., 1., 0., 0., 0.]])
    assert np.all(np.abs(rbt - relu_back_test) < 1e-20), "relu gradient incorrect!"

    linear_back_test, bias_back_test = linear_backward(X, w, b)
    print("linear_back_test")
    print(linear_back_test)
    lbt = np.array([[-0.38314889,  0.35891731,  0.09037955,  0.98397352, -0.74292248, -0.5883056,  0.54354937,  0.79001348],
                    [ 0.58758113, -0.412598  ,  0.08740239, -0.68723605, -0.29251551, 0.36521658, -0.25330565,  0.03919754],
                    [ 0.97960327, -0.41368028,  0.26308195,  0.94303171, -0.92383705, 0.28187289,  0.35914219, -0.46526478],
                    [ 0.2583081 ,  0.97956892,  0.31049517, -0.68557195, -0.68612885, -0.9054485, -0.70507179,  0.11431403],
                    [-0.7674351 ,  0.69421738, -0.8007104 ,  0.93470719,  0.61132148, 0.54328029,  0.00919623, -0.34544161]])
    assert np.all(np.abs(lbt - linear_back_test) < 1e-10), "Linear gradient incorrect!"
    print("bias")
    print(bias_back_test)   # should be 1

    print("All tests passed!")