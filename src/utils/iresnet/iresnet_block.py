import torch.nn as nn

from src.utils.iresnet.spectral_norm_fc import spectral_norm_fc

class iresnet_block(nn.Module):
    def __init__(self, in_features, int_features=10, coeff=.97, n_power_iter=5, nonlin="elu"):
        """
        build invertible block
        :param in_shape: shape of the input (channels)
        :param int_dims: dimension of intermediate layers
        :param coeff: desired lipschitz constant
        :param n_power_iter: number of iterations for spectral normalization
        :param nonlin: the nonlinearity to use
        """
        super(iresnet_block, self).__init__()
        self.coeff = coeff
        self.n_power_iter = n_power_iter
        nonlin = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus
        }[nonlin]

        layers = []

        layers.append(spectral_norm_fc(nn.Linear(in_features, int_features, bias=True), self.coeff,
                                       n_power_iterations=self.n_power_iter))
        layers.append(nonlin())
        layers.append(spectral_norm_fc(nn.Linear(int_features, int_features, bias=False), self.coeff,
                                       n_power_iterations=self.n_power_iter))
        layers.append(nonlin())
        layers.append(spectral_norm_fc(nn.Linear(int_features, in_features, bias=False), self.coeff,
                                       n_power_iterations=self.n_power_iter))

        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective or injective block forward """

        Fx = self.bottleneck_block(x)

        # add residual to output
        y = Fx + x
        return y

    def inverse(self, y, maxIter=100):
        # inversion of ResNet-block (fixed-point iteration)
        x = y
        for iter_index in range(maxIter):
            summand = self.bottleneck_block(x)
            x = y - summand
        return x