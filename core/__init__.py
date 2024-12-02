import torch
from torch import nn
import yaml
import os

import core.metrics
import core.scheduler
import core.data
import core.loss
import core.model
import core.optim

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
        self.model = core.model.get_model(model, **self.model_config)

    def forward(self, x):
        return self.model(x)
    # 载入数据， 数据预处理
    def train_model(self, config):
        config = self._load_config(config)
        self.train_config = config["train"]
        self.val_config = config["val"]


        train_data = core.data.get_dataset(**self.train_config["dataset"])
        train_dataloader = core.data.get_dataloader(train_data, **self.train_config["dataloader"])
        loss_fn = core.loss.get_loss_fn(self.train_config["loss_fn"])

        optimizer = core.optim.get_optimizer(self.model.parameters(), self.train_config["optimizer"], **self.train_config["optimizer_args"])
        # if hasattr(optimizer, config['lr_scheduler']):
        #     lr_scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler'])
        # else:
        #     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=getattr(core.scheduler, config['lr_scheduler']))

        for epoch in range(self.train_config['epochs']):
            for i, (data, label) in enumerate(train_dataloader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()
                # lr_scheduler.step()
                print(loss)

    def val_model(self):
        pass
    @staticmethod
    def _load_config(file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File path: {file_path} Not Found!")
        with open(file_path, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return data