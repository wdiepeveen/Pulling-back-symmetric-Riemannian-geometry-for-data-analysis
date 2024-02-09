import torch

def hyperbolic(n_points, noise=0.1):
    """
     Returns the two spirals dataset.
    """
    n = torch.linspace(-2.,2.,n_points)[:,None]
    d1x = torch.sinh(n)
    d1y = torch.cosh(n)
    return torch.hstack((d1x,d1y)) + torch.randn((n_points,2)) * noise