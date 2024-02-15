import torch

from src.manifolds import Manifold

class Hyperboloid(Manifold):
    """ Base class describing the unit hyperboloid in dimension `d+1` """

    def __init__(self, d):
        super().__init__(d)

        self.M = torch.eye(self.d + 1)
        self.M[self.d,self.d] = -1.

    def inner(self, p, X, Y):
        """

        :param p: N x (d + 1)
        :param X: N x M x (d + 1)
        :param Y: N x L x (d + 1)
        :return: N x M x L
        """
        assert (len(p.shape) + 1) == len(X.shape) == len(Y.shape)
        assert p.shape[-1] == X.shape[-1] == Y.shape[-1] == self.d + 1 and p.shape[:-1] == X.shape[:-2] == Y.shape[:-2]
        if len(p.shape) > 2:  # so N is a tensor
            pp = p.reshape(-1, p.shape[-1])
            XX = X.reshape(-1, X.shape[-2], X.shape[-1])
            YY = Y.reshape(-1, Y.shape[-2], Y.shape[-1])
            return self.inner(pp, XX, YY).reshape(p.shape[:-1], X.shape[-2], Y.shape[-2])
        else:
            return self.minkowski_metric(X, Y)

    def distance(self, p, q):
        """

        :param p: N x M x (d + 1)
        :param q: N x L x (d + 1)
        :return: N x M x L
        """
        assert len(p.shape) == len(q.shape)
        assert p.shape[-1] == q.shape[-1] == self.d + 1 and p.shape[:-2] == q.shape[:-2]
        return torch.arccosh(torch.clamp(-torch.einsum("NMi,NLj,ij->NML", p, q, self.M),1.))

    def log(self, p, q):
        """

        :param p: N x (d + 1)
        :param q: N x M x (d + 1)
        :return: N x M x (d + 1)
        """
        assert (len(p.shape) + 1) == len(q.shape)
        X = q + self.minkowski_metric(p.unsqueeze(-2), q).squeeze(-2).unsqueeze(-1) * p.unsqueeze(-2)
        d_pq = self.distance(q, p[:,None])
        return d_pq * X / (self.norm(p, X).unsqueeze(-1) + 1e-8)

    def exp(self, p, X):
        """

        :param p: N x (d + 1)
        :param X: N x M x (d + 1)
        :return: N x M x (d + 1)
        """
        assert (len(p.shape) + 1) == len(X.shape)
        X_norm = self.norm(p, X).unsqueeze(-1) + 1e-8
        return torch.cosh(X_norm) * p.unsqueeze(-2) + torch.sinh(X_norm) * X / X_norm

    def parallel_transport(self, p, X, q):
        """

        :param p: N x (d + 1)
        :param X: N x M x (d + 1)
        :param q: N x (d + 1)
        :return: N x M x (d + 1)
        """
        return X - torch.einsum("NMi,NLj,ij->NML", X, self.log(p, q.unsqueeze(-2)), self.M) / (self.distance(p.unsqueeze(-2),q.unsqueeze(-2))**2 + 1e-8) * (self.log(p,q.unsqueeze(-2)) + self.log(q,p.unsqueeze(-2)))

    def manifold_dimension(self):
        return self.d

    def minkowski_metric(self, X, Y):
        """

        :param p: N x (d + 1)
        :param X: N x M x (d + 1)
        :param Y: N x L x (d + 1)
        :return: N x M x L
        """
        return torch.einsum("NMi,NLj,ij->NML", X, Y, self.M)

