import torch

def twospirals(n_points, noise=.5, num_turns=1):
    """
     Returns the two spirals dataset.
    """
    n = torch.sqrt(torch.linspace(1/n_points,1.,n_points)[:,None]) * 360 * num_turns * (2*torch.pi)/360
    d1x = -torch.cos(n)*n + torch.rand(n_points,1) * noise
    d1y = torch.sin(n)*n + torch.rand(n_points,1) * noise - noise
    return torch.vstack((torch.hstack((d1x,d1y)),torch.hstack((-d1x,-d1y)), torch.zeros(1,2)))