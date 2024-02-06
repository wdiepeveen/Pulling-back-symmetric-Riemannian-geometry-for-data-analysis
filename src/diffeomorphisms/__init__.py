import torch.nn as nn

class Diffeomorphism(nn.Module): # TODO we have 5 arguments for the construction: 1) z, 2) O, 3) phi_theta, 4) psi, 5) manifold and the class should implement the differentials and stuff
    """ Base class describing a diffeomorphism phi: R^d \to M """

    def __init__(self, embedding_manifold):
        super(Diffeomorphism, self).__init__()
        self.manifold = embedding_manifold
        self.d = self.manifold.d
        self.diffeo = None

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
