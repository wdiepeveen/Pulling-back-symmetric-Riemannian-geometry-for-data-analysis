import torch.nn as nn

class Diffeomorphism(nn.Module):
    """ Base class describing a diffeomorphism varphi: R^d \to M  of the form varphi = psi^{-1} o phi o O o T_z"""

    def __init__(self, embedding_manifold, offset=None, orthogonal=None, deformation=None, chart=None):
        """

        :param embedding_manifold: d-dimensional Manifold
        :param offset: d tensor
        :param orthogonal: d x d tensor
        :param deformation: SimpleDiffeomorphism
        :param chart: SimpleDiffeomorphism
        """
        super(Diffeomorphism, self).__init__()
        self.M = embedding_manifold
        self.d = self.M.d

        self.z = offset
        self.O = orthogonal
        self.phi = deformation
        self.psi = chart

    def forward(self, x):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def inverse(self, p):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def differential_forward(self, x, X):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def differential_inverse(self, p, P):
        raise NotImplementedError(
            "Subclasses should implement this"
        )
