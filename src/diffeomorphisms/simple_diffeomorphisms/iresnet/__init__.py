import torch.nn as nn

from src.diffeomorphisms.simple_diffeomorphisms import SimpleDiffeomorphism
from src.diffeomorphisms.simple_diffeomorphisms.iresnet.iresnet_block import iresnet_block

class i_ResNet(SimpleDiffeomorphism):
    def __init__(self, in_features, nBlocks=50, int_features=10, coeff=.97, n_power_iter=5, nonlin="elu"):
        super(i_ResNet, self).__init__()

        self.nBlocks = nBlocks
        self.i_resnet_layers = nn.ModuleList([iresnet_block(in_features,
                                                            int_features=int_features,
                                                            coeff=coeff,
                                                            n_power_iter=n_power_iter,
                                                            nonlin=nonlin) for block in range(self.nBlocks)])

    def forward(self, y):
        """ iresnet forward """
        x = y
        for block in self.i_resnet_layers:
            x = block(x)
        return x

    def inverse(self, y, maxIter=100):
        """ iresnet inverse """
        x = y
        for block in reversed(self.i_resnet_layers):
            x = block.inverse(x, maxIter=maxIter)
        return x
