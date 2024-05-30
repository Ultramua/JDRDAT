import torch
from torch.utils.data import Dataset


class eegDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor, z_tensor):
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor

        assert self.x.size(0) == self.y.size(0)
        assert self.z.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]

    def __len__(self):
        return len(self.y)
