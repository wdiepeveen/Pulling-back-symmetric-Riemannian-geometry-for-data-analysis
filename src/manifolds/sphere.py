import torch

from src.manifolds import Manifold

class Sphere(Manifold):
    """ Base class describing the unit sphere in dimension `d+1` """

    def __init__(self, d):
        super().__init__(d)

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
            return torch.einsum("NMi,NLi->NML", X, Y)

    # def norm(self, p, X):
    #     """
    #
    #     :param p: N x (d + 1)
    #     :param X: N x M x (d + 1)
    #     :param Y: N x L x (d + 1)
    #     :return: N x M x L
    #     """
    #     return torch.sqrt(self.inner(p.unsqueeze(-2) * torch.ones((1,X.shape[-2],1)), X.unsqueeze(-2),X.unsqueeze(-2)).squeeze(-2))

    def distance(self, p, q):
        """

        :param p: N x M x (d + 1)
        :param q: N x L x (d + 1)
        :return: N x M x L
        """
        assert len(p.shape) == len(q.shape)
        assert p.shape[-1] == q.shape[-1] == self.d + 1 and p.shape[:-2] == q.shape[:-2]
        return torch.arccos(torch.clamp(torch.einsum("NMi,NLi->NML", p, q),-1.,1.))

    def log(self, p, q):
        """

        :param p: N x (d + 1)
        :param q: N x M x (d + 1)
        :return: N x M x (d + 1)
        """
        assert (len(p.shape) + 1) == len(q.shape)
        X = q - torch.einsum("Ni,NMi,Nj->NMj", p, q, p)
        d_pq = self.distance(q, p[:,None])
        return d_pq * X / (torch.norm(X,2,-1).unsqueeze(-1) + 1e-8)

    def exp(self, p, X):
        """

        :param p: N x (d + 1)
        :param X: N x M x (d + 1)
        :return: N x M x (d + 1)
        """
        assert (len(p.shape) + 1) == len(X.shape)
        X_norm = torch.norm(X,2,-1).unsqueeze(-1) + 1e-8
        y = torch.cos(X_norm) * p.unsqueeze(-2) + torch.sin(X_norm) * X / X_norm
        return y / torch.norm(y,2,-1).unsqueeze(-1)

    def parallel_transport(self, p, X, q):
        """

        :param p: N x (d + 1)
        :param X: N x M x (d + 1)
        :param q: N x (d + 1)
        :return: N x M x (d + 1)
        """
        return X - torch.einsum("NMi,NLi->NML", X, self.log(p, q.unsqueeze(-2))) / (self.distance(p.unsqueeze(-2),q.unsqueeze(-2))**2 + 1e-8) * (self.log(p,q.unsqueeze(-2)) + self.log(q,p.unsqueeze(-2)))

    def manifold_dimension(self):
        return self.d
