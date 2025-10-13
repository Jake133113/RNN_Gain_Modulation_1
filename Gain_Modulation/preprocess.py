import numpy as np

'''Synthetic Inputs: create random initial neural responses to test the whitening process. '''

def iso_gaussian(N, batch):  #batch of vectors that should already have variances of 1
    return np.random.randn(batch, N)

def diag_anisotropic(N, batch, rng=None): #batch of vectors that has different variances (output variances should all be 1 after whitening)
    rng = np.random.default_rng() if rng is None else rng
    sig2 = 10**rng.uniform(-1, 1, size=N)     # variances spanning 0.1..10
    X = rng.normal(size=(batch, N)) * np.sqrt(sig2)[None, :]
    return X

def correlated_gaussian(N, batch, rng=None):  # Manually creating vectors from a correlated covariance matrix
    rng = np.random.default_rng() if rng is None else rng
    Q, _ = np.linalg.qr(rng.normal(size=(N, N)))     # random orthonormal
    lam = 10**rng.uniform(-1, 1, size=N)             # eigenvalues
    L = Q @ (np.sqrt(lam)[:, None] * Q.T)            # Σ^{1/2}
    X = rng.normal(size=(batch, N)) @ L.T
    return X

def center(vec):
    mean = np.mean(vec)
    new_vec = [(i - mean) for i in vec]
    return new_vec

def build_projection(D, N, seed=42):
    """Return P ∈ R^{N x D} with row-normalized Gaussian entries."""
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(N, D))
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    return P

# ---- Optional helper: convert to single-channel if needed ----
def to_grayscale(x):
    """
    x: (..., H, W) or (..., H, W, C). If C==3/4 convert via simple luminance,
    if C==1 return as-is.
    """
    if x.ndim >= 3 and x.shape[-1] in (3, 4):
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        return 0.2989*r + 0.5870*g + 0.1140*b
    return x  # already single-channel

# ---- Main: images → centered N-D inputs ----
def images_to_inputs(batch_imgs, P, scale_uint8=True):
    """
    batch_imgs: np.ndarray of shape (B, H, W) or (B, H, W, C), all same size.
    P: projection matrix (N×D), where D = H*W after grayscale.
    Returns S: (B, N) with per-image mean removed.
    """
    X = batch_imgs.astype(np.float64, copy=False)
    if scale_uint8 and X.dtype == np.uint8:
        X = X / 255.0

    # grayscale if needed
    X = to_grayscale(X)                  # (B, H, W)
    assert X.ndim == 3, "Expect (B,H,W) after grayscale."

    B, H, W = X.shape
    D = H * W
    N = P.shape[0]
    assert P.shape == (N, D), f"P must be (N={N}, D={D}) for images of size {H}×{W}"

    # flatten and per-image mean center
    V = X.reshape(B, D)
    V = V - V.mean(axis=1, keepdims=True)

    # project to N dims
    S = V @ P.T                           # (B, N)
    return S.astype(np.float64)