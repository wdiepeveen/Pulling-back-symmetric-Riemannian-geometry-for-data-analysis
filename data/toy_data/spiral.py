import torch


def twospirals(n_points, noise=.5, num_turns=1, t0=0.):
    """
     Returns the two spirals dataset.
    """
    t_offset = torch.tensor([t0])
    n = torch.sqrt(torch.linspace(1 / (2 * n_points), 1., n_points)[:, None]) * 360 * num_turns * (2 * torch.pi) / 360 + t_offset
    d1x = -torch.cos(n) * n + torch.cos(t_offset) * t_offset + torch.randn(n_points, 1) * noise
    d1y = torch.sin(n) * n - torch.sin(t_offset) * t_offset + torch.randn(n_points, 1) * noise
    return torch.vstack((torch.hstack((d1x, d1y)).flip(0), torch.zeros(1, 2), torch.hstack((-d1x, -d1y))))
