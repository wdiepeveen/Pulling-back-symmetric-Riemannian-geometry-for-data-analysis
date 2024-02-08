import torch
import torch.autograd.forward_ad as fwAD

from src.diffeomorphisms import Diffeomorphism

class Vanilla_into_Manifold(Diffeomorphism):
    def __init__(self, manifold, offset, orthogonal, chart):
        super().__init__(manifold, offset=offset, orthogonal=orthogonal, chart=chart)

    def forward(self, x):
        """
        :param x: N x d or N x M x d
        :return: N x dd or N x M x dd
        """
        if len(x.shape) == 2:
            fwd = self.psi.inverse(torch.einsum("ij,Nj->Ni", self.O, x - self.z[None]))
        elif len(x.shape) == 3:
            fwd = self.psi.inverse(torch.einsum("ij,NMj->NMi", self.O, x - self.z[None, None]))
        else:
            raise NotImplementedError(
                "len(x.shape) is not 2 nor 3"
            )
        return fwd

    def inverse(self, p):
        """
        :param p: N x dd or N x M x dd
        :return: N x d or N x M x d
        """
        if len(p.shape) == 2:
            inv = torch.einsum("ij,Nj->Ni", self.O.T, self.psi.forward(p)) + self.z[None]
        elif len(p.shape) == 3:
            inv = torch.einsum("ij,NMj->NMi", self.O.T, self.psi.forward(p)) + self.z[None, None]
        else:
            raise NotImplementedError(
                "len(x.shape) is not 2 nor 3"
            )
        return inv

    def differential_forward(self, x, X):
        """
        :param x: N x d
        :param X: N x M x d
        :return: N x M x dd
        """
        primal = x.clone().requires_grad_()
        L = X.shape[-2]
        tangents = X

        output = torch.zeros((*tangents.shape[:-1], self.d + 1)).T
        with fwAD.dual_level():
            for l in range(L):
                tangent = tangents.T[:, l].T
                dual_input = fwAD.make_dual(primal, tangent)
                dual_output = self.forward(dual_input)
                differential = fwAD.unpack_dual(dual_output).tangent
                output[:, l] = differential.T
        D_x_forward_X = output.T

        return D_x_forward_X

    def differential_inverse(self, p, P):
        """
        :param p: N x dd
        :param P: N x M x dd
        :return: N x M x d
        """
        primal = p.clone().requires_grad_()
        tangents = P
        L = tangents.shape[-2]

        output = torch.zeros((*tangents.shape[:-1], self.d)).T
        with fwAD.dual_level():
            for l in range(L):
                tangent = tangents.T[:, l].T
                dual_input = fwAD.make_dual(primal, tangent)
                dual_output = self.inverse(dual_input)
                differential = fwAD.unpack_dual(dual_output).tangent
                output[:, l] = differential.T
        D_p_inverse_P = output.T

        return D_p_inverse_P