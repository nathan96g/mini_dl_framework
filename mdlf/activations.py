from mdlf.module import Module
from torch import empty

class Activation(Module):
    
    def __init__(self, dim=-1):
        self.input_dim = dim
        self.output_dim = dim
        self.number_params = 0
        self.input = None

    def forward(self, input):
        raise NotImplementedError('forward')
    
    def backward(self, *grad_from_output):
        raise NotImplementedError('backward')
    
    def initialize(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = input_dim
    
    def __str__(self):
        return "Activation"

class ReLU(Activation):
    #store z_l : z_l -> activation(z_l) -> x_l
    def forward(self, input):
        #pk self.input n'est pas égale à input.where(input >= 0, empty(input.shape).fill_(0)) ?
        self.input = input #
        return input.where(input >= 0, empty(input.shape).fill_(0))
    
    #compute activation_deriv(z_l)
    #TODO : think if the delta{L+1} is computed componentwise !!! => if yes can greatly simplify
    def backward(self, *grad_wrt_output):
        activ_eval = (self.input >= 0).to(self.input.dtype)
        if len(grad_wrt_output) == 0 :
            return activ_eval
        else :
            a = grad_wrt_output[0] * activ_eval
            return grad_wrt_output[0] * activ_eval
    
    def __str__(self):
        return super().__str__() + ": ReLU"

class Tanh(Activation):

    def forward(self, input):
        self.input = input.tanh()
        return self.input
    
    def backward(self, *grad_wrt_output):
        derivative_Tanh = (1 - (self.input.tanh() ** 2)).to(self.input.dtype)
        if len(grad_wrt_output) == 0 :
            return derivative_Tanh
        else :
            return grad_wrt_output[0] * derivative_Tanh
    
    def __str__(self):
        return super().__str__() +": Tanh"

class Identity(Activation):
    def forward(self, input):
        return input
    
    #TODO: Check if right shape
    def backward(self, *grad_wrt_output):
        if len(grad_wrt_output) == 0 :
            # change !!!
            return empty((1)).fill_(1) 
        else :
            return grad_wrt_output[0] * empty((grad_wrt_output[0].size())).fill_(1)
    
    def __str__(self):
        return super().__str__() + ": Identity"


