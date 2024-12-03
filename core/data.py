import custom.dataset
import custom.preprocess

from custom.dataset import *
from torch.utils.data import DataLoader, Dataset



def get_dataset(data: str, mode: str = 'train'):
    if os.path.exists(data):
        if mode == ["train", "valid", "test"]:
            return ImageLabelDataset(os.path.join(data, mode, "images"), os.path.join(data, mode, "labels"))
        else:
            raise ValueError(f"模式填错了，你填成了：{mode}")
    elif hasattr(custom.dataset, data):
        return getattr(custom.dataset, data)(train=mode)
    else:
        raise ValueError(f"没有这种数据集，请你重新确定数据集！")

def get_dataloader(dataset: Dataset, shuffle: bool, batch_size: int, num_workers: int, collate_fn=None):
    DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    return DataLoader(dataset)