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
    """
    The ReLU function is ReLU(x) = max(0,x)
    applied component-wise.
    """

    #store z_l : z_l -> activation(z_l) -> x_l
    def forward(self, input):
        self.input = input 
        return self.input.relu()
        #return input.where(input >= 0, empty(input.shape).fill_(0))
    
    #compute activation_deriv(z_l)
    def backward(self, *grad_wrt_output):
        derivative = (self.input >= 0).to(self.input.dtype)        
        return grad_wrt_output[0] * derivative
    
    def __str__(self):
        return super().__str__() + ": ReLU"

class Tanh(Activation):
    """
    The Tanh function is Tanh(x) = (exp(2x)-1) / (exp(2x)+1)
    applied component-wise.
    """

    def __init__(self, dim=-1):
        super().__init__(dim)
        self.tanh = None # avoid recomputation in backward pass

    def forward(self, input):
        self.input = input
        self.tanh = input.tanh()
        return self.tanh
        # e = (2*input).exp()
        # return (e-1)/(e+1)
    
    def backward(self, *grad_wrt_output):
        derivative_Tanh = (4 * (self.tanh.exp() + self.tanh.mul(-1).exp()).pow(-2)).to(self.input.dtype)
        return grad_wrt_output[0] * derivative_Tanh
    
    def __str__(self):
        return super().__str__() +": Tanh"

class Sigmoid(Activation):
    """
    The Sigmoid function is Sigmoid(x) = 1 / (1+exp(-x))
    applied component-wise.
    """

    def __init__(self, dim=-1):
        super().__init__(dim)
        self.sigmoid = None # avoid recomputation in backward pass

    def forward(self, input):
        self.input = input
        self.sigmoid = input.sigmoid()
        return self.sigmoid
    
    def backward(self, *grad_wrt_output):
        derivative = 1-self.sigmoid
        return grad_wrt_output[0] * derivative
    
    def __str__(self):
        return super().__str__() +": Sigmoid"

class Identity(Activation):
    """
    The Identity function is Identity(x) = x
    applied component-wise.
    """

    def forward(self, input):
        self.input = input
        return input
    
    def backward(self, *grad_wrt_output):
        return grad_wrt_output[0] # * [1,1...,1] which is unnecessary
    
    def __str__(self):
        return super().__str__() + ": Identity"


