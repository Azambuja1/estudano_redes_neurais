from torch.utils.data import Dataset, DataLoader
import torch.distributions.uniform as urand

class AlgebraicDataSet(Dataset):
    def __init__(self, f, interval, nsamples):
        X = urand.Uniform(interval[0],interval[1]).sample(nsamples)
        self.data = [ (x, f(x)) for x in X]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]