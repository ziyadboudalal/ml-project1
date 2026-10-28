import numpy as np


def compute_mse(y, tx, w):
    """Compute half the mean squared error."""
    error = y - tx.dot(w)
    return 0.5 * np.mean(error**2)


def compute_mse_gradient(y, tx, w):
    """Compute the MSE gradient for the provided samples."""
    error = y - tx.dot(w)
    return -tx.T.dot(error) / len(y)


def compute_logistic_loss(y, tx, w):
    """Compute mean binary cross-entropy for labels in {0, 1}."""
    scores = tx.dot(w)

    # Compute the loss stably for each target class.
    signed_scores = (1 - 2 * y) * scores
    return np.mean(np.logaddexp(0.0, signed_scores))


def compute_logistic_gradient(y, tx, w):
    """Compute the mean binary cross-entropy gradient."""
    scores = tx.dot(w)

    # Compute sigmoid probabilities without exponential overflow.
    probabilities = np.exp(-np.logaddexp(0.0, -scores))

    return tx.T.dot(probabilities - y) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: Target values, shape (N,).
        tx: Feature matrix, shape (N, D).
        initial_w: Initial weights, shape (D,).
        max_iters: Number of gradient descent updates.
        gamma: Learning rate.

    Returns:
        w: Final weights, shape (D,).
        loss: Half the mean squared error at the final weights.
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        gradient = compute_mse_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = compute_mse(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent

    Args:
        y: Target values, shape (N,).
        tx: Feature matrix, shape (N, D).
        initial_w: Initial weights, shape (D,).
        max_iters: Number of stochastic gradient descent updates.
        gamma: Learning rate.

    Returns:
        w: Final weights, shape (D,).
        loss: Half the mean squared error over all training samples.
    """
    w = initial_w.copy()
    N = len(y)

    for _ in range(max_iters):
        index = np.random.randint(N)
        y_batch = y[index : index + 1]
        tx_batch = tx[index : index + 1]

        gradient = compute_mse_gradient(y_batch, tx_batch, w)
        w = w - gamma * gradient

    loss = compute_mse(y, tx, w)

    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations

    Args:
        y: Target values, shape (N,).
        tx: Feature matrix, shape (N, D).

    Returns:
        w: Optimal weights, shape (D,).
        loss: Half the mean squared error at the returned weights.
    """

    # Form the normal equations in double precision even for float32 features.
    # One-hot columns and missingness flags can be linearly dependent. In that
    # case, solve the normal equations with their Moore-Penrose inverse rather
    # than returning enormous, unstable coefficients from a near-singular solve.
    x64 = np.asarray(tx, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    a = x64.T.dot(x64)
    b = x64.T.dot(y64)
    eigenvalues = np.linalg.eigvalsh(a)
    cutoff = np.finfo(float).eps * max(a.shape)
    if eigenvalues[0] <= cutoff * max(eigenvalues[-1], 1.0):
        w = np.linalg.pinv(a, rcond=cutoff).dot(b)
    else:
        w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations

    Args:
        y: Target values, shape (N,).
        tx: Feature matrix, shape (N, D).
        lambda_: Nonnegative regularization parameter.

    Returns:
        w: Optimal regularized weights, shape (D,).
        loss: Half the mean squared error, excluding the penalty.
    """
    N, D = tx.shape

    a = tx.T.dot(tx) + 2 * N * lambda_ * np.eye(D)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y ∈ {0, 1})

    Args:
        y: Binary target values in {0, 1}, shape (N,).
        tx: Feature matrix, shape (N, D).
        initial_w: Initial weights, shape (D,).
        max_iters: Number of gradient descent updates.
        gamma: Learning rate.

    Returns:
        w: Final weights, shape (D,).
        loss: Mean binary cross-entropy at the final weights.
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = compute_logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Fit logistic regression with L2 regularization.

    Args:
        y: Binary target values in {0, 1}, shape (N,).
        tx: Feature matrix, shape (N, D).
        lambda_: Nonnegative regularization parameter.
        initial_w: Initial weights, shape (D,).
        max_iters: Number of gradient descent updates.
        gamma: Learning rate.

    Returns:
        w: Final weights, shape (D,).
        loss: Mean binary cross-entropy, excluding the penalty.
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        gradient = gradient + 2 * lambda_ * w

        w = w - gamma * gradient

    loss = compute_logistic_loss(y, tx, w)

    return w, loss
