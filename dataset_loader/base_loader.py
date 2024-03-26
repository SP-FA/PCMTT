from torch.utils.data import Dataset


class BaseLoader(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError