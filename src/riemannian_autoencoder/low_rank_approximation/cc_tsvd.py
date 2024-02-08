import torch

from src.riemannian_autoencoder.low_rank_approximation.naive_tsvd import naive_low_rank_approximation

def curvature_corrected_low_rank_approximation(M, x, X, rank):
    """

        :param M: Manifold
        :param x: Mpoint
        :param X: N x Mpoint
        :param rank: int
        :return: rank x Tvector, rank x N
        """

    n = X.shape[0]
    d = M.d
    r = min(n, d, rank)

    # compute initialisation
    R_x, U = naive_low_rank_approximation(M, x, X, r)
    # construct linear system
    log_x_X = M.log(x[None], X[None])  # ∈ T_x M^n

    # construct matrix βκB

    # U: r x N
    # phi_x: D x Tvector
    # theta_x: N x D x Tvector
    # beta: N x D

    inners = M.inner(x[None], Phi_x[None], beta_kappa[:,:,None] * Theta_x)[0] # N x D # TODO we need to repeat x and Phi_x along axis 0

    tensor_bkB = torch.einsum("NjDi", U, inners) #tensorU. * inner.(Ref(M), Ref(q), tensorΨq, tensorβκΘq)

    βκB = reshape(tensorβκB, (n[1] * d, r * d))

    # construct matrix A
    A = transpose(βκB) * βκB

    # construct vector βκBb
    b = reshape(tensorb, (n[1] * d))
    βκBb = transpose(βκB) * b

    # solve linear system
    Vₖₗ = A\βκBb
    tensorVₖₗ = reshape(Vₖₗ, (r, d))

    # get ccRr_q
    ccR_q = get_vector.(Ref(M), Ref(q), [tensorVₖₗ[l, :] for l = 1:r], Ref(DefaultOrthonormalBasis()))
    return ccR_q, U
