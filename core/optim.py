import torch.optim

def get_optimizer(parameters, optimizer_name, **kargs):
    return getattr(torch.optim, optimizer_name)(parameters, **kargs)
