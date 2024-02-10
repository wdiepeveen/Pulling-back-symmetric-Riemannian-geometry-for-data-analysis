import torch
import torch.autograd.forward_ad as fwAD

from src.diffeomorphisms import Diffeomorphism
from src.manifolds.product_manifold import ProductManifold
from src.manifolds.euclidean import Euclidean
from src.diffeomorphisms.simple_diffeomorphisms.iresnet import i_ResNet

class i_ResNet_into_Euclidean(Diffeomorphism):
    def __init__(self, d, offset, orthogonal, nBlocks=50, max_iter_inverse=100, int_features=10, coeff=.97,
                 n_power_iter=5):
        super().__init__(ProductManifold([Euclidean(d[0]), Euclidean(d[1])]), offset=offset, orthogonal=orthogonal)

        self.d0 = d[0]
        self.d1 = d[1]

        self.phi = i_ResNet(self.d, nBlocks=nBlocks, int_features=int_features, coeff=coeff, n_power_iter=n_power_iter)
        self.max_iter_inverse = max_iter_inverse

    def forward(self, x, asproduct=True):
        """
        :param x: N x d or N x M x d
        :return: [N x d0, N x d1] or [N x M x d0, N x M x d1]
        """
        if len(x.shape) == 2:
            fwd = self.phi(torch.einsum("ij,Nj->Ni", self.O, x - self.z[None]))
        elif len(x.shape) == 3:
            fwd = self.phi(torch.einsum("ij,NMj->NMi", self.O, x - self.z[None,None]))
        else:
            raise NotImplementedError(
                "len(x.shape) is not 2 nor 3"
            )
        if asproduct:
            return [fwd.T.split([self.d0, self.d1])[0].T, fwd.T.split([self.d0, self.d1])[1].T]
        else:
            return fwd

    def inverse(self, p):
        """
        :param p: [N x d0, N x d1]
        :return: N x d
        """
        if len(p[0].shape) == len(p[1].shape) == 2:
            inv = torch.einsum("ij,Nj->Ni", self.O.T, self.phi.inverse(torch.cat(p, -1), maxIter=self.max_iter_inverse)) + self.z[None]
        elif len(p[0].shape) == len(p[1].shape) == 3:
            inv = torch.einsum("ij,NMj->NMi", self.O.T, self.phi.inverse(torch.cat(p, -1), maxIter=self.max_iter_inverse)) + self.z[None,None]
        else:
            raise NotImplementedError(
                "len(x.shape) is not 2 nor 3"
            )
        return inv

    def differential_forward(self, x, X):
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
                dual_output = self.forward(dual_input, asproduct=False)
                differential = fwAD.unpack_dual(dual_output).tangent
                output[:, l] = differential.T
        D_x_forward_X = output.T

        return [D_x_forward_X.T.split([self.d0, self.d1])[0].T, D_x_forward_X.T.split([self.d0, self.d1])[1].T]

    def differential_inverse(self, p, P):
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
                dual_output = self.inverse(dual_input)
                differential = fwAD.unpack_dual(dual_output).tangent
                output[:, l] = differential.T
        D_p_inverse_P = output.T

        return D_p_inverse_P