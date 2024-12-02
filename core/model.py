import torch
from torch import nn
import yaml
import models

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
        config = yaml.safe_load(open(config, 'r'))

        self.model = getattr(models, model)(**config)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _load_config(file_path: str) -> dict:
        with open(file_path, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return data