import numpy as np


def eval_err(r_t): #input is a matrix of the whitening output. r_t vectors should be colmum-stacked
    N = int(np.shape(r_t)[0])
    I = np.eye(N)
    cov = np.cov(r_t)
    w = np.linalg.eigvalsh(cov-I)
    error = float(np.max(np.abs(w)))
    return error

