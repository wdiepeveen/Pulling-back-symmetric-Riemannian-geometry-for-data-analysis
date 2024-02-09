import torch

from src.manifolds import Manifold

class ProductManifold(Manifold):
    """ Base class describing a product manifold M1 x ... x Mn """

    def __init__(self, manifolds):
        assert type(manifolds) == list
        assert len(manifolds) > 1
        self.n = len(manifolds)
        self.manifolds = manifolds

        d = 0
        di = []
        for i in range(self.n):
            di.append(self.manifolds[i].d)
            d += self.manifolds[i].d

        super().__init__(d)
        self.di = di

    def inner(self, p, X, Y):
        """

        :param p: [N x d1, ..., N x dn]
        :param X: [N x M x d1, ..., N x M x dn]
        :param Y: [N x L x d1, ..., N x L x dn]
        :return: N x M x L
        """
        for i in range(self.n):
            assert (len(p[i].shape) + 1) == len(X[i].shape) == len(Y[i].shape)
            assert p[i].shape[-1] == X[i].shape[-1] == Y[i].shape[-1] == self.di[i]
            assert p[i].shape[:-1] == X[i].shape[:-2] == Y[i].shape[:-2]
        if len(p[0].shape) > 2:  # so N is a tensor
            pp = []
            XX = []
            YY = []
            for i in range(self.n):  # TODO we have to account for that the Mpoint can have shape len > 1
                pp.append(p[i].reshape(-1, self.di[i]))
                XX.append(X[i].reshape(-1, X.shape[-2], X.shape[-1]))
                YY.append(Y[i].reshape(-1, Y.shape[-2], Y.shape[-1]))
            return self.inner(pp, XX, YY).reshape(p[0].shape[:-1], X[0].shape[-2], Y[0].shape[-2])
        else:
            return sum([self.manifolds[i].inner(p[i], X[i], Y[i]) for i in range(self.n)])

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
        return [self.manifolds[i].geodesic(p[i], q[i], t) for i in range(self.n)]

    def distance(self, p, q):
        """

        :param p: [N x M x d1, ..., N x M x dn]
        :param q: [N x L x d1, ..., N x L x dn]
        :return: N x M x L
        """
        for i in range(self.n):
            assert len(p[i].shape) == len(q[i].shape)
            assert p[i].shape[-1] == q[i].shape[-1] == self.di[i] and p[i].shape[:-2] == q[i].shape[:-2]
        return torch.sqrt(sum([self.manifolds[i].distance(p[i], q[i]) ** 2 for i in range(self.n)]))

    def log(self, p, q):
        """

        :param p: [N x d1, ..., N x dn]
        :param q: [N x M x d1, ..., N x M x dn]
        :return: [N x M x d1, ..., N x M x dn]
        """
        for i in range(self.n):
            assert (len(p[i].shape) + 1) == len(q[i].shape)
        return [self.manifolds[i].log(p[i], q[i]) for i in range(self.n)]

    def exp(self, p, X):
        """

        :param p: [N x d1, ..., N x dn]
        :param X: [N x M x d1, ..., N x M x dn]
        :return: [N x M x d1, ..., N x M x dn]
        """
        for i in range(self.n):
            assert (len(p[i].shape) + 1) == len(X[i].shape)
        return [self.manifolds[i].exp(p[i], X[i]) for i in range(self.n)]

    def parallel_transport(self, p, X, q):
        return [self.manifolds[i].parallel_transport(p[i], X[i], q[i]) for i in range(self.n)]

    def manifold_dimension(self):
        return self.d
