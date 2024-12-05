import math


def default(**kwargs)->dict:
    return {}

def batch_status(**kwargs)-> dict:
    print(f"epoch: {kwargs['epoch']} | batch: {kwargs['i']} / {math.ceil(len(kwargs['train_data']) / kwargs['dataloader']['batch_size'])} | loss: {kwargs['loss']}")
    return {}

def epoch_status(**kwargs)-> dict:
    if kwargs['epoch'] % 1 == 0:
        print(f"epoch: {kwargs['epoch']} | loss: {kwargs['loss']}")
    return {}