from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. (3073x10)
    - X (the training array): A numpy array of shape (N, D) containing a minibatch of data. (1000, 3073)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means (1000,)
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    - sccore: W*X has shape (N,C) ; a label for each input image n
    
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero (3073,10)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        
        image_xi = X[i,:] #3073 size
        scores = image_xi.dot(W) #10 size
        
        # remember that this is the training set
        correct_class_score = scores[y[i]]
        count_over_margin = 0

        for j in range(num_classes):
            
            if j != y[i]:
                margin = scores[j] - correct_class_score + 1  # note delta = 1
            else: 
                margin = -1
            
            if margin > 0:
                count_over_margin += 1
                loss += margin
                
                #gradient column
                
                dwj = image_xi
                dW[:, j] += dwj
                
        dW[:,y[i]]  -= count_over_margin * X[i,:]
            

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train (N).
    loss /= num_train
    dW = dW/num_train
    
    #adding the regularization term
    dW = dW + 2 * reg * W
    # Add regularization to the loss.
    loss += reg * np.sum(W * W) #multiply each term by itself w11 * w11

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    num_train = X.shape[0]
    num_labels = W.shape[1]
    num_pixels = X.shape[1]
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero (3073,10)
    
    
    scores_vector = np.zeros((num_train, num_labels)) #(500,10)
    scores_vector = np.dot(X, W) 
    
    correct_class_scores = scores_vector[np.arange(num_train), y]
    
    scores_vector = scores_vector - np.reshape(correct_class_scores, (num_train, 1)) + 1 #(500,10)
    margin = np.sum(np.maximum(scores_vector, 0), axis = 1) - 1 #(500,) 500 correct classes
    
    #gradient for j = i
    mask = scores_vector > 0 #(500,10)
    count_over_margin = np.sum(mask, axis = 1) - 1 #(500,)
    
   
    #the -1 kills the additional term from the correct class score, it's the same as:
    #margin[np.arange(num_train), y] = 0
    data_loss = sum(margin)
    
    #averaging
    data_loss = data_loss/num_train
    loss = data_loss + reg * np.sum(W * W)
    
    
    #values matrix is one for j != yi or count_over_margin for yi
    values = mask * 1.0
    
    
    values[np.arange(num_train),y] =  -count_over_margin
    
    #X.T (3073, 500) @ (500, 10): shape (3073, 10) == dW shape 
    dWdata = (X.T).dot(values) / num_train
    
    #adding the regularization term
    dW = dWdata + 2 * reg * W
    
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
