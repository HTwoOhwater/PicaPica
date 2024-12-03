import torch
import custom.loss_fn

def get_loss_fn(loss_name: str):
    if hasattr(torch.nn, loss_name):
        loss_fn = getattr(torch.nn, loss_name)()
        return loss_fn
    elif hasattr(custom.loss_fn, loss_name):
        loss_fn = getattr(custom.loss_fn, loss_name)()
        return loss_fn()
    else:
        raise ValueError(f"损失函数名 {loss_name} 不存在！")
