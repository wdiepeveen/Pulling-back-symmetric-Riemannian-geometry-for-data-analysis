import torch

from src.diffeomorphisms.simple_diffeomorphisms import SimpleDiffeomorphism

class StereographicHyperboloidChart(SimpleDiffeomorphism):
    """ The stereographic projection chart psi: H^d \to R^d  """
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, p):
        """
        :param p: N x (d + 1) or N x M x (d + 1)
        :return: N x d or N x M x d
        """
        output = p.T[:-1]
        x = output.T
        return x

    def inverse(self, x):
        """
        :param x: N x d or N x M x d
        :return: N x (d + 1) or N x M x (d + 1)
        """
        y = x
        s_sq = torch.sum(y ** 2, -1)

        output = torch.zeros((*y.shape[:-1], self.d + 1)).T
        output[:-1] = y.T
        output[-1] = torch.sqrt(s_sq.T + 1)

        p = output.T
        return p
