from mdlf.module import Module
from torch import empty

class Layer(Module):
    def __str__(self):
        return "Layer"

    def param(self): 
        raise NotImplementedError('param')

    def update(self, new_weights):
        raise NotImplementedError('update')

class Identity(Layer):
    #TODO: Implement it : do identity
    def forward(self, input):
        return input
    
    def backward(self, grad_wrt_output):
        raise NotImplementedError('backward')
    
    def param(self): 
        raise NotImplementedError('param')

    def update(self, *new_weights):
        raise NotImplementedError('update')




class Linear(Layer):

    def __init__(self, number_nodes, input_dim=-1):
        self.output_dim = number_nodes
        self.input_dim = input_dim
        self.number_params = -1
        self.weights = None
        self.bias = None
        self.input = None
        self.weights_grad = None
        self.bias_grad = None

# see if output needed
# if we are in layer l:
# input : x_{l-1} 
# output : z_l
    def forward(self, input):
        self.input = input
        return self.weights.T @ input + self.bias
    
    def backward(self, *grad_wrt_output):
        curr_delta = grad_wrt_output[0] #delta_l
        self.weights_grad = self.input.unsqueeze(1) @ curr_delta.unsqueeze(0)
        self.bias_grad = curr_delta     
        prev_delta_partial = (self.weights @ curr_delta) # delta_{l-1} without componentwise activation mult
        
        return prev_delta_partial 

    def param(self): 
        return [(self.weights, self.weights_grad), (self.bias, self.bias_grad)]

    def update(self, new_weights):
        self.weights = new_weights[0]
        self.bias = new_weights[1]
    
    def initialize(self, input_dim):

        # Handle automatic dimension initialization
        if input_dim == -1:
            if self.input_dim == -1:
                raise RuntimeError("This dense layer need a manual initialization "
                                   "of the input dimension, e.g."
                                   " Linear(number_nodes=150, input_dim=64)")
        else:
            if self.input_dim != -1 and input_dim != self.input_dim:
                raise RuntimeError("The manually set input dimension ({}) of this layer"
                                   " is different from the one inferred automatically ({})"
                                   " by the initialization of the sequential model".format(self.input_dim, input_dim))
            else:
                self.input_dim = input_dim

        # Initialize weights and bias : TODO : initiate with correct uniform
        self.weights = empty(self.input_dim, self.output_dim).uniform_(0, 1)
        self.bias = empty(self.output_dim).fill_(0)
        self.number_params = self.output_dim * self.input_dim + self.output_dim
    
    def __str__(self):
        return super().__str__() + ": Linear" 

#TODO
class Conv1D(Layer):

    def initialize(self, input_dim):
        raise NotImplementedError

    def forward(self, *input): 
        raise NotImplementedError

    def backward(self, *gradwrtoutput): 
        raise NotImplementedError

    def param(self): 
        raise NotImplementedError('param')
    
    def update(self, *new_weights):
        raise NotImplementedError('update')

    def __str__(self):
        return super().__str__() + ": Convolution 1D" 