import torch
from torch.utils.data import Dataset

class DistanceData(Dataset):
    def __init__(self, data, pairwise_distances):
        assert len(pairwise_distances.shape) == 2
        assert data.shape[0] == pairwise_distances.shape[0] == pairwise_distances.shape[1]
        self.data = data
        self.N = pairwise_distances.shape[0]
        i = torch.tensor([k for k in range(self.N)])[:, None].repeat(1, self.N)
        j = torch.tensor([k for k in range(self.N)])[None, :].repeat(self.N, 1)

        selected_entries = (j != i) * (pairwise_distances != float("inf"))

        self.i = i[selected_entries]
        self.j = j[selected_entries]
        self.pairwise_distances = pairwise_distances[i[selected_entries], j[selected_entries]]

    def __len__(self):
        return len(self.i)

    def __getitem__(self, idx):
        return self.data[self.i[idx]], self.data[self.j[idx]], self.pairwise_distances[idx]