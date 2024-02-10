import torch

from src.manifolds import Manifold

class PullBackManifold(Manifold):
    """ Base class describing a learned manifold of dimension `d` """

    def __init__(self, diffeo):
        self.manifold = diffeo.M
        self.diffeo = diffeo
        super().__init__(self.manifold.d)

    def inner(self, p, X, Y):
        return self.manifold.inner(self.diffeo.forward(p),
                                   self.diffeo.differential_forward(p, X),
                                   self.diffeo.differential_forward(p, Y)
                                   )

    def geodesic(self, p, q, t):
        return self.diffeo.inverse(self.manifold.geodesic(self.diffeo.forward(p), self.diffeo.forward(q), t))

    def distance(self, p, q):
        return self.manifold.distance(self.diffeo.forward(p), self.diffeo.forward(q))

    def log(self, p, q):
        return self.diffeo.differential_inverse(self.diffeo.forward(p),
                                                self.manifold.log(self.diffeo.forward(p), self.diffeo.forward(q))
                                                )

    def exp(self, p, X):
        return self.diffeo.inverse(self.manifold.exp(self.diffeo.forward(p), self.diffeo.differential_forward(p, X)))

    def parallel_transport(self, p, X, q):
        return self.diffeo.differential_inverse(self.diffeo.forward(q),
                                                self.manifold.parallel_transport(self.diffeo.forward(p),
                                                                                 self.diffeo.differential_forward(p, X),
                                                                                 self.diffeo.forward(q)
                                                                                 )
                                                )

    def manifold_dimension(self):
        return self.d

    def metric_tensor_in_std_basis(self, p):
        """

        :param p: N x d
        :return: N x d x d
        """
        N = p.shape[0]
        return self.inner(p, torch.eye(self.d,self.d)[None].repeat(N, 1, 1), torch.eye(self.d,self.d)[None].repeat(N, 1, 1))
