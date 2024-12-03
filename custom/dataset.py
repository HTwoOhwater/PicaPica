import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets import MNIST
import os
from torchvision import transforms


def mnist(root='./data', train=True):
    return MNIST(root, train, download=True, transform=transforms.ToTensor())


class ImageLabelDataset(Dataset):
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path

        self.images = []
        self.labels = []

        image_list = os.listdir(self.image_path)
        label_list = os.listdir(self.label_path)

        for image, label in (image_list, label_list):
            image = Image.open(os.path.join(self.image_path, image))
            image.convert('RGB')
            image = np.array(image)
            self.images.append(image)

            with open(os.path.join(self.label_path, label), 'r') as f:
                lines = []
                for line in f.readlines():
                    lines.append(line.strip('\n').strip(''))
                lines = np.array(lines)
                self.labels.append(lines)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.labels = torch.from_numpy(self.labels)
        self.images = torch.from_numpy(self.images)

    def __len__(self):
        assert len(self.images) == len(self.labels), "图像数据和标注数据不匹配"
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]
