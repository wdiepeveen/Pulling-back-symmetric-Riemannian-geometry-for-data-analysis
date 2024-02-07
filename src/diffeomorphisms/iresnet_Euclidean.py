import torch
import torch.autograd.forward_ad as fwAD

from src.diffeomorphisms import Diffeomorphism
from src.manifolds.product_manifold import ProductManifold
from src.manifolds.euclidean import Euclidean
from src.utils.iresnet import i_ResNet

class i_ResNet_into_Euclidean(Diffeomorphism):
    def __init__(self, d, offset, orthogonal, nBlocks=50, max_iter_inverse=100, int_features=10, coeff=.97,
                 n_power_iter=5):
        super().__init__(offset, orthogonal, None, None, ProductManifold([Euclidean(d[0]), Euclidean(d[1])]))

        self.d0 = d[0]
        self.d1 = d[1]

        self.phi = i_ResNet(self.d, nBlocks=nBlocks, int_features=int_features, coeff=coeff, n_power_iter=n_power_iter)
        self.max_iter_inverse = max_iter_inverse

    def forward(self, x):  # TODO
        """
        :param x: N x d
        :return: [N x d0, N x d1]
        """
        fwd = self.net(x - self.offset)
        return [fwd.T.split([self.d0, self.d1])[0].T, fwd.T.split([self.d0, self.d1])[1].T]

    def inverse(self, p):  # TODO
        """
        :param p: [N x d0, N x d1]
        :return: N x d
        """
        return self.net.inverse(torch.cat(p, -1), maxIter=self.max_iter_inverse) + self.offset

    def differential_forward(self, x, X):  # TODO
        """
        :param x: N x d
        :param X: N x M x d
        :return: N x M x d
        """
        primal = x.clone().requires_grad_()
        L = X.shape[-2]
        tangents = X

        output = torch.zeros(X.shape).T
        with fwAD.dual_level():
            for l in range(L):
                tangent = tangents.T[:, l].T
                dual_input = fwAD.make_dual(primal, tangent)
                dual_output = self.net.forward(dual_input)
                differential = fwAD.unpack_dual(dual_output).tangent
                output[:, l] = differential.T
        D_x_forward_X = output.T

        return [D_x_forward_X.T.split([self.d0, self.d1])[0].T, D_x_forward_X.T.split([self.d0, self.d1])[1].T]

    def differential_inverse(self, p, P):  # TODO
        """
        :param p: [N x d0, N x d1]
        :param P: [N x M x d0, N x M x d1]
        :return: [N x M x d0, N x M x d1]
        """
        primal = torch.cat(p, -1).clone().requires_grad_()
        tangents = torch.cat(P, -1)
        L = tangents.shape[-2]

        output = torch.zeros(tangents.shape).T
        with fwAD.dual_level():
            for l in range(L):
                tangent = tangents.T[:, l].T
                dual_input = fwAD.make_dual(primal, tangent)
                dual_output = self.net.inverse(dual_input, maxIter=self.max_iter_inverse)
                differential = fwAD.unpack_dual(dual_output).tangent
                output[:, l] = differential.T
        D_p_inverse_P = output.T

        return D_p_inverse_P