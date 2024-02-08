import torch

class Curvature_Corrected_Riemannian_Autoencoder:
    def __init__(self, pullback_manifold, base_point, basis):
        self.M = pullback_manifold

        assert base_point.shape == (self.M.d,)
        self.z = base_point

        assert len(basis.shape) == 2 and basis.shape[0] <= self.M.d and basis.shape[1] == self.M.d
        self.w_z = basis
        self.k = basis.shape[0]

    def encode(self, x):
        """
        :param x: N x d tensor
        :return : N x k tensor
        """
        log_z_x = self.M.log(self.z[None], x[None])
        latent_coefficients = self.M.inner(self.z[None], log_z_x, self.w_z[:, None])[0]
        return latent_coefficients

    def decode(self, a):
        """
        :param a: N x k tensor
        :return : N x d tensor
        """
        Xi_z = torch.einsum("Nk,kd->Nd", a, self.w_z)
        exp_Xi_z = self.M.exp(self.z[None], Xi_z[None])[0]
        return exp_Xi_z

    def project_on_manifold(self, x):
        """
        :param x: N x d tensor
        :return : N x d tensor
        """
        return self.decode(self.encode(x))