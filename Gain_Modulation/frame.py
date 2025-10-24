"""
frame.py — initializes a fixed matrix of synaptic weights W, the gain matrix diag(g),
and the gradient step size gamma, wrapped in a Frame class.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Optional


class Frame:
    """
    Frame:
      - N: Dimension of input 
      - K: minimal overcomplete basis size = N*(N+1)/2
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
    ) -> None:
        """
        Args:
            dim: dimension of input
            gamma: gradient step size; defaults to 5e-3 if None
            init_weights: if True, build W during init
            init_gain: if True, build g during init
        """
        self.dim = int(dim)
        self.K = np.int64(self.dim * (self.dim + 1) // 2)

        # Parameters
        self.gamma = gamma if gamma is not None else 5e-3

        # Initialize matrices
        self.W: Optional[np.ndarray] = None
        self.g: Optional[np.ndarray] = None

        if init_weights:
            self.W = self.mercedes()          
        if init_gain:
            self.g = np.ones(self.K)      

    def mercedes(self) -> torch.Tensor:
        """
        Build an overcomplete set of K unit vectors (columns) in N-dim using a
        greedy "least-cosine-to-selected" selection from random candidates.

        """
        N, K = self.dim, self.K

        #generate random normalized vectors (columns of matrix A)
        A = np.random.randn(4 * K, N)
        A /= np.linalg.norm(A, axis=1, keepdims=True)

        # Start with one random vector (first row), build W whose columns are selected vectors
        w_1 = A[0]
        A = np.delete(A, 0, axis=0)
        W = np.stack([w_1], axis=1)  # shape (N, 1)

        # Pick the candidate whose max abs cosine to current W is minimal
        for _ in range(K - 1):
            cos = A @ W                      # shape (3K-1 - t, t+1)
            closest = np.max(np.abs(cos), axis=1)
            idx = np.argmin(closest)
            W = np.column_stack([W, A[idx]])  # append column
            A = np.delete(A, idx, axis=0)
        return W