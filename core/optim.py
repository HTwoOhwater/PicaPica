import torch.optim

def get_optimizer(parameters, optimizer_name, **kwargs):
    return getattr(torch.optim, optimizer_name)(parameters, **kwargs)
