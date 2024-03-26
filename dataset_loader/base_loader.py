from torch.utils.data import Dataset

from dataset_util.base_class import BaseDataset


class BaseLoader(Dataset):
    def __init__(self, data: BaseDataset):
        self.data = data
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError