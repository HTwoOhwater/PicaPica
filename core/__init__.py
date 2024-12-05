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
import core.callbacks

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
        callbacks = core.callbacks.get_callbacks(**self.train_config["callbacks"])

        train_data = core.data.get_dataset(**self.train_config["dataset"])
        train_dataloader = core.data.get_dataloader(train_data, **self.train_config["dataloader"])
        loss_fn = core.loss.get_loss_fn(self.train_config["loss_fn"])

        optimizer = core.optim.get_optimizer(self.model.parameters(), self.train_config["optimizer"], **self.train_config["optimizer_args"])
        running_dict = {"train_config": self.train_config,
                        "train_dataloader": train_dataloader,
                        "train_data": train_data,
                        "loss_fn": loss_fn,
                        "optimizer": optimizer
                        }
        # if hasattr(optimizer, config['lr_scheduler']):
        #     lr_scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler'])
        # else:
        #     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=getattr(core.scheduler, config['lr_scheduler']))
        train_start = {**running_dict, **callbacks["train_start"](**running_dict)}
        self.model.train()
        for epoch in range(self.train_config['epochs']):
            epoch_end = {}
            epoch_start = {**train_start, **{"epoch": epoch}, **callbacks["epoch_start"](**train_start, **{"epoch": epoch})}
            for i, (data, label) in enumerate(train_dataloader):
                batch_start = {**epoch_start, **{"i": i, "data": data, "label": label}, **callbacks["batch_start"](**epoch_start, **{"i": i, "data": data, "label": label})}
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()
                # lr_scheduler.step()
                batch_end = {**batch_start, **{"loss": loss, "output": output}, **callbacks["batch_end"](**batch_start, **{"loss": loss, "output": output})}
            epoch_end = {**epoch_start, **callbacks["epoch_end"](**batch_end)}
        train_end = {**train_start, **callbacks["train_end"](**epoch_end)}
    def val_model(self):
        pass
    @staticmethod
    def _load_config(file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File path: {file_path} Not Found!")
        with open(file_path, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return data