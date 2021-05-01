from mdlf.module import Module
import torch

class Activation(Module):
    
    def __init__(self, dim=-1):
        self.input_dim = dim
        self.output_dim = dim
        self.number_params = 0

    def forward(self, input):
        raise NotImplementedError('forward')
    
    def backward(self, grad_from_output):
        raise NotImplementedError('backward')
    
    def initialize(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = input_dim
    
    def __str__(self):
        return "Activation"

class ReLU(Activation):

    def forward(self, input):
        return torch.relu(input)
    
    def backward(self, grad_wrt_output):
        return None
    
    def __str__(self):
        return super().__str__() + ": ReLU"

class Tanh(Activation):

    def forward(self, input):
        return torch.tanh(input)
    
    def backward(self, grad_wrt_output):
        return None
    
    def __str__(self):
        return super().__str__() +": Tanh"


class Identity(Activation):
    def forward(self, input):
        return input
    
    #TODO: Check if right shape
    def backward(self, *grad_wrt_output):
        return torch.empty((1)).fill_(1)
    
    def __str__(self):
        return ""


