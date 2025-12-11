"""
frame.py — Initializes a fixed matrix of synaptic weights W, the diagonal gain matrix diag(g),
and the gradient step size (gamma)

"""

from __future__ import annotations
import numpy as np
from typing import Optional


class Frame:
    """
    Frame:
      - N: Dimension of input 
      - K: minimal overcomplete basis size >= N*(N+1)/2
      - W: (N x K) weight matrix
      - g: (K x K) diagonal gain matrix 
      - gamma: gradient step size 
    """

    def __init__(
        self,
        dim: int,
        gamma: Optional[float] = None,
        init_weights: bool = True,
        init_gain: bool = True,
        multi_timescale: bool = False,
    ) -> None:
        """
        Args:
            dim: dimension of input vector
            gamma: gradient step size; defaults to 5e-3 if None
            init_weights: if True, build W during init
            init_gain: if True, build g during init
            multi_timescale: if false, frame dim scales as O(N^2), if true O(N) instead
        """
        self.dim = int(dim)
        if multi_timescale == False:
            self.K = np.int64(self.dim * (self.dim + 1) // 2)
            self.gamma = gamma if gamma is not None else 5e-3
        else:
            self.K = np.int64(2*self.dim)
            self.gamma_r = 5e-3
            self.gamma_g = 5e-2
            self.gamma_w = 1e-5

        # Parameters
        self.gamma = gamma if gamma is not None else 5e-3

        # Initialize matrices
        self.W: Optional[np.ndarray] = None
        self.g: Optional[np.ndarray] = None

        if init_weights:
            self.W = self.mercedes()          
        if init_gain:
            self.g = np.ones(self.K)      

    def mercedes(self) -> np.ndarray:
        """
        Build an overcomplete set of K unit vectors (columns) in N-dim using a
        greedy "least-cosine-to-selected" selection from random candidates. In 2D, 
        this resembles a Mercedes Benz logo, hence the name.

        """
        N, K = self.dim, self.K

        # Generate random normalized vectors that will be selected from (columns of a matrix A)
        A = np.random.randn(5 * K, N) # The 5 here is arbitrary, just need much more than 
        A /= np.linalg.norm(A, axis=1, keepdims=True)

        # Start with one random vector (first row), build W whose columns are selected vectors
        w_1 = A[0]
        A = np.delete(A, 0, axis=0)
        W = np.stack([w_1], axis=1)  # shape (N, 1)

        # Pick the candidate whose max abs dot product to current W is minimal
        for _ in range(K - 1):
            cos = A @ W                      # shape (3K-1 - t, t+1)
            closest = np.max(np.abs(cos), axis=1)
            idx = np.argmin(closest)
            W = np.column_stack([W, A[idx]])  # append column
            A = np.delete(A, idx, axis=0)

        W /= np.linalg.norm(W, axis=0, keepdims=True)

        return W