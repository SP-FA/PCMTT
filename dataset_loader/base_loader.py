from torch.utils.data import Dataset
import numpy as np

from dataset_util.base_class import BaseDataset


class BaseLoader(Dataset):
    def __init__(self, data: BaseDataset, cfg):
        self.data = data
        self.cfg = cfg
        self.gaussian = self.Gaussian()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    class Gaussian:
        def __init__(self):
            self.mean = np.zeros(3)
            self.cov = np.diag([1, 1, 5])

        def sample(self, n):
            return np.random.multivariate_normal(self.mean, self.cov, size=n)
