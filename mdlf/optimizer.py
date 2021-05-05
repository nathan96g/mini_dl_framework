import torch
from torch import empty


class Optimizer:
    def step(model, train_data, train_label):
        return NotImplementedError

class SGD(Optimizer):
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def step(model, train_data, train_label):
        
        #1. sample randomly one
        train_sample = None
        train_sample_label = None

        model.backward(train_sample, train_sample_label)
        #TODO: iterate over all layers and update their weights with their gradient computed previously
        # for module in model.modules:
            # if instanceof(layer):
            #     module.param += lambda_ * module.grad

        return model