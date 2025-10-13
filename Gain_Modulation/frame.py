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
    Frame encapsulates:
      - N: Dimension of input 
      - K: minimal overcomplete basis size = N*(N+1)/2
      - W: (N x K) weight matrix (torch.Tensor on chosen device)
      - g: (K x K) diagonal gain matrix (numpy.ndarray, identity by default)
      - gamma: gradient step size (float)
      - device: "cuda" if available and requested, else "cpu"
    """

    def __init__(
        self,
        N: int,
        gamma: Optional[float] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        init_weights: bool = True,
        init_gain: bool = True,
    ) -> None:
        """
        Args:
            N: dimension of input
            gamma: gradient step size; defaults to 1e-2 if None
            seed: NumPy RNG seed for reproducibility of W construction
            device: "cuda" / "cpu" / None (auto)
            init_weights: if True, build W during init
            init_gain: if True, build g during init
        """
        self.N = int(N)
        self.K = np.int64(self.N * (self.N + 1) // 2)

        # RNG seed (affects only NumPy operations here)
        if seed is not None:
            np.random.seed(seed)

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Parameters
        self.gamma = gamma if gamma is not None else self.starting_gamma()

        # Initialize matrices
        self.W: Optional[torch.Tensor] = None
        self.g: Optional[np.ndarray] = None

        if init_weights:
            self.W = self.mercedes()          # (N x K) on self.device
        if init_gain:
            self.g = self.starting_g()        # (K x K) numpy.identity

    # --------------------------
    # Methods corresponding to your original functions
    # --------------------------

    def mercedes(self) -> torch.Tensor:
        """
        Build an overcomplete set of K unit vectors (columns) in R^N using a
        greedy "least-cosine-to-selected" selection from 3K random candidates.

        Returns:
            W_torch: (N x K) torch.Tensor on self.device
        """
        N, K = self.N, self.K

        # 3K candidate rows in R^N, then L2-normalize rows
        A = np.random.randn(3 * K, N)
        A /= np.linalg.norm(A, axis=1, keepdims=True)

        # Start with one random vector (first row), build W whose columns are selected vectors
        w_1 = A[0]
        A = np.delete(A, 0, axis=0)
        W = np.stack([w_1], axis=1)  # shape (N, 1)

        # Greedily add K-1 vectors: pick the candidate whose max abs cosine to current W is minimal
        for _ in range(K - 1):
            cos = A @ W                      # shape (3K-1 - t, t+1)
            closest = np.max(np.abs(cos), axis=1)
            idx = np.argmin(closest)
            W = np.column_stack([W, A[idx]])  # append column
            A = np.delete(A, idx, axis=0)

        # Convert to torch tensor (should not be needed here)
        W_torch = torch.from_numpy(W).float().to(self.device)
        return W

    def starting_g(self) -> np.ndarray:
        """
        Initialize diag(g) as an identity (K x K).
        """
        return np.ones(self.K)

    def starting_gamma(self) -> float:
        """
        Default gradient step size.
        """
        return 1e-2