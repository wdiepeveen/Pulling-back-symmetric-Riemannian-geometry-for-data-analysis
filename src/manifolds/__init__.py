import torch

class Manifold:
    """ Base class describing a manifold of dimension `ndim` """

    def __init__(self, d):
        self.d = d

    def barycentre(self, x, tol=1e-3, max_iter=20):
        """

        :param x: N x M x Mpoint
        :return: N x Mpoint
        """
        k = 0
        rel_error = 1.
        y = x[:,0]
        while k <= max_iter and rel_error >= tol:
            y = self.exp(y, torch.mean(self.log(y, x),1).unsqueeze(-2)).squeeze(-2)
            k+=1

        return y

    def inner(self, p, X, Y):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def norm(self, p, X):
        """

        :param p: N x Mpoint
        :param X: N x M x Mpoint
        :return: N x M
        """
        return torch.sqrt(self.inner(p.unsqueeze(-2) * torch.ones((1, X.shape[-2], 1)),
                                     X.unsqueeze(-2), X.unsqueeze(-2)).squeeze(-2))

    def distance(self, p, q):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def log(self, p, q):
        """

        :param p: N x Mpoint
        :param q: N x M x Mpoint
        :return: N x M x Mpoint
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def exp(self, p, X):
        """

        :param p: N x Mpoint
        :param X: N x M x Mpoint
        :return: N x M x Mpoint
        """
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def geodesic(self, p, q, t):
        """

        :param p: 1 x Mpoint
        :param q: 1 x Mpoint
        :param t: N
        :return: N x Mpoint
        """
        assert p.shape[0] == q.shape[0] == 1
        assert len(t.shape) == 1
        return self.exp(p, t.unsqueeze(0).unsqueeze(2) * self.log(p, q.unsqueeze(1)))[0]

    def parallel_transport(self, p, X, q):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def manifold_dimension(self):
        return self.d
