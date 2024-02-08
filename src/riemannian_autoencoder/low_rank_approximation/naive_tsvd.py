import torch

def naive_low_rank_approximation(M, x, X, rank):
    """

    :param M: Manifold
    :param x: Mpoint
    :param X: N x Mpoint
    :param rank: int
    :return: rank x Mpoint, rank x N
    """
    n = X.shape[0]
    # compute log
    log_x_X = M.log(x[None], X[None])[0]  # ∈ T_x M^n
    # tangent space SVD
    r = min(n, M.d, rank)
    # compute Gram matrix
    Gram_mat = M.inner(x[None], log_x_X[None], log_x_X[None])[0]
    # compute U and R_x
    L, U = torch.linalg.eigh(Gram_mat)
    R_x = torch.einsum("Nk,Nd->kd", U[:, -r:], log_x_X)
    return R_x, U[:, -r:].T