from mdlf.module import Module
from torch import empty

class Activation(Module):
    
    def __init__(self, dim=-1):
        self.input_dim = dim
        self.output_dim = dim
        self.number_params = 0
        self.output = None

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
        output = input.where(input >= 0,  empty(input.shape).fill_(0))
        self.output = output
        return output
    
    def backward(self, *grad_wrt_output):
        return None
    
    def __str__(self):
        return super().__str__() + ": ReLU"

#TODO
class Tanh(Activation):

    def forward(self, input):
        return NotImplementedError('forward')
    
    def backward(self, grad_wrt_output):
        return None
    
    def __str__(self):
        return super().__str__() +": Tanh"

#TODO : implement activation that changes nothing
class Identity(Activation):
    def forward(self, input):
        return NotImplementedError('forward')
        # return input
    
    #TODO: Check if right shape
    def backward(self, *grad_wrt_output):
        return empty((1)).fill_(1)
    
    def __str__(self):
        return ""


