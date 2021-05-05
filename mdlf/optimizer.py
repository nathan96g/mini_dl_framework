import torch
from torch import empty


class Optimizer:
    def step(model, train_data, train_label):
        return NotImplementedError

class SGD(Optimizer):
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def step(model, train_data, train_label):
        #TODO: iterate over all layers and update their weights with their gradient computed previously
        train_sample = torch.empty(0)
        train_sample_label = 
        return NotImplementedError