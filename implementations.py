import numpy as np

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Return final weights and full-data MSE using single-sample SGD."""
    # TODO:
    raise NotImplementedError("Implement mean_squared_error_sgd")


def least_squares(y, tx):
    """Return optimal weights and MSE using the normal equations."""
    # TODO:
    raise NotImplementedError("Implement least_squares")


def ridge_regression(y, tx, lambda_):
    """Return regularized weights and MSE, excluding the L2 penalty."""
    # TODO:
    raise NotImplementedError("Implement ridge_regression")


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Return final weights and mean binary cross-entropy."""
    # TODO:
    raise NotImplementedError("Implement logistic_regression")


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Return final weights and mean binary cross-entropy without penalty."""
    # TODO:
    raise NotImplementedError("Implement reg_logistic_regression")