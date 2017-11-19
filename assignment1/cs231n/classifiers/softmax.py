import numpy as np
from random import shuffle
from past.builtins import xrange
from math import log, exp 

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_trains = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_trains):
    f = X[i].dot(W)
    f -= np.max(f)
    fi_sum = 0
    dW2 = np.zeros_like(W)
    for j in xrange(num_classes):
      fi_sum += exp(f[j])
      dW2[:, j] += X[i]*exp(f[j])
    loss += -f[y[i]] + log(fi_sum)
    dW2 /= fi_sum
    dW += dW2
    dW[:, y[i]] += -X[i]

  loss /= num_trains
  dW /= num_trains

  loss += reg*np.sum(W**2)
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W)
  scores -= scores.max(axis=1)[:, np.newaxis]
  scores_exp = np.exp(scores)
  sum_scores = np.sum(scores_exp, axis=1)
  loss_i = scores_exp[np.arange(N), y]/sum_scores

  loss = np.sum(-np.log(loss_i))/N
  loss += reg * np.sum(W**2)

  dSy = np.zeros_like(scores)
  dSy[np.arange(N), y] = 1
  dS = scores_exp / sum_scores[:, np.newaxis] - dSy

  dW = X.T.dot(dS)

  dW /= N
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

