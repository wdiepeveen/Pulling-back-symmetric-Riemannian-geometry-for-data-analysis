import torch.nn as nn

class Diffeomorphism(nn.Module):
    """ Base class describing a diffeomorphism varphi: R^d \to M  of the form varphi = psi^{-1} o phi o O o T_z"""

    def __init__(self, offset, orthogonal, deformation, chart, embedding_manifold):
        super(Diffeomorphism, self).__init__()
        self.z = offset
        self.O = orthogonal
        self.phi = deformation
        self.psi_inverse = chart
        self.M = embedding_manifold
        self.d = self.M.d

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
