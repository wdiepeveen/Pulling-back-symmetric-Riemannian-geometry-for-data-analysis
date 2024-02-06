import torch

from src.manifolds import Manifold

class Euclidean(Manifold):
    """ Base class describing Euclidean space of dimension `d` """

    def __init__(self, d, a=1.):
        super().__init__(d)
        self.a = a

    def inner(self, p, X, Y):
        """

        :param p: N x d
        :param X: N x M x d
        :param Y: N x L x d
        :return: N x M x L
        """
        assert (len(p.shape) + 1) == len(X.shape) == len(Y.shape)
        assert p.shape[-1] == X.shape[-1] == Y.shape[-1] == self.d and p.shape[:-1] == X.shape[:-2] == Y.shape[:-2]
        if len(p.shape) > 2:  # so N is a tensor
            pp = p.reshape(-1, self.d)
            XX = X.reshape(-1, X.shape[-2], X.shape[-1])
            YY = Y.reshape(-1, Y.shape[-2], Y.shape[-1])
            return self.inner(pp, XX, YY).reshape(p.shape[:-1], X.shape[-2], Y.shape[-2])
        else:
            return self.a * torch.einsum("NMi,NLi->NML", X, Y)

    # def sharp(self, p, Xi):
    #     return Xi

    # def flat(self, p, X):
    #     return X

    def geodesic(self, p, q, t):
        """

        :param p: 1 x Mpoint
        :param q: 1 x Mpoint
        :param t: N x float
        :return: N x Mpoint
        """
        return (1 - t[:,None]) * p[None] + t[:,None] * q[None]

    def distance(self, p, q):
        """

        :param p: N x M x d
        :param q: N x L x d
        :return: N x M x L
        """
        assert len(p.shape) == len(q.shape)
        assert p.shape[-1] == q.shape[-1] == self.d and p.shape[:-2] == q.shape[:-2]
        return torch.sqrt(self.a * torch.sum((p.unsqueeze(-2) - q.unsqueeze(-3)) ** 2, -1) + 1e-8)

    def log(self, p, q):
        """

        :param p: N x d
        :param q: N x M x d
        :return: N x M x d
        """
        assert (len(p.shape) + 1) == len(q.shape)
        return q - p.unsqueeze(-2)

    def exp(self, p, X):
        """

        :param p: N x d
        :param X: N x M x d
        :return: N x M x d
        """
        assert (len(p.shape) + 1) == len(X.shape)
        return p.unsqueeze(-2) + X

    def parallel_transport(self, p, X, q):
        """

        :param p: N x d
        :param X: N x M x d
        :param q: N x d
        :return: N x M x d
        """
        return X

    def manifold_dimension(self):
        return self.d
