import torch

def circle(n_points, noise=0.1):
    """
     Returns the two spirals dataset.
    """
    n = torch.linspace(0.1,0.9,n_points)[:,None] * 2*torch.pi
    d1x = -torch.cos(n)
    d1y = torch.sin(n)
    return torch.hstack((d1x,d1y)) + torch.randn((n_points,2)) * noise