import torch
from torch import nn
import yaml
from torch.nn.modules.module import T
import os
import models
import core

# 我想要一些模型，给定模型一些要求，我就能流畅地获得模型的结果。

class Model(nn.Module):
    """
    实现功能：
        根据字符串名称实例化我想要的模型
        根据数据要求对我的数据做加载和处理
        根据训练选项得到
        支持断点继续训练
    """
    def __init__(self, model: str, config: str):
        super(Model, self).__init__()
        # 先载入我们的信息
        config = self._load_config(config)
        self.model_config = config
        self.model = getattr(models, model)(**self.model_config)

    def forward(self, x):
        return self.model(x)
    # 载入数据， 数据预处理
    def train_model(self, config):
        config = self._load_config(config)
        self.train_config = config

        CustomDataSet = getattr(core.data, self.train_config["data"])
        train_data = CustomDataSet()

        CustonDataLoader = getattr(core.data, self.train_config["dataloader"])
        train_dataloader =

        optimizer = getattr(torch.optim, config['optimizer'])
        if hasattr(optimizer, config['lr_scheduler']):
            lr_scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler'])
        else:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=getattr(core.scheduler, config['lr_scheduler']))

        for epoch in range(self.train_config['epochs']):
            for i, data, label in enumerate(self.train_loader):
                output = self.model(data)
        pass

    def val_model(self):
        pass
    @staticmethod
    def _load_config(file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File path: {file_path} Not Found!")
        with open(file_path, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return data