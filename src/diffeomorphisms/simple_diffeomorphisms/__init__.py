import torch.nn as nn

class SimpleDiffeomorphism(nn.Module):
    """ Base class describing a diffeomorphism varphi: M \to N """

    def __init__(self):
        super(SimpleDiffeomorphism, self).__init__()

    def forward(self, x):
        raise NotImplementedError(
            "Subclasses should implement this"
        )

    def inverse(self, p):
        raise NotImplementedError(
            "Subclasses should implement this"
        )
