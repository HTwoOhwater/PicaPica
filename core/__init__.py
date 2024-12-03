import torch
from torch import nn
import yaml
import os

import core.metrics
import core.scheduler
import core.data
import core.loss

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
        self.model = get_model

    def forward(self, x):
        return self.model(x)
    # 载入数据， 数据预处理
    def train_model(self, config):
        config = self._load_config(config)
        self.train_config = config["train"]
        self.val_config = config["val"]


        train_data =
        CustomDataLoader = getattr(core.data, self.train_config["dataloader"])
        train_dataloader = CustomDataLoader(train_data, self.train_config["batch_size"])

        loss_fn = getattr(core.loss, self.train_config["loss_fn"])()

        optimizer = getattr(torch.optim, self.train_config['optimizer'])(self.model.parameters())
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