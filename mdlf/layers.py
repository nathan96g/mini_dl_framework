from mdlf.module import Module
from torch import empty

class Layer(Module):
    def __str__(self):
        return "Layer"

class Identity(Layer):
    #TODO: Implement it : do identity
    def forward(self, input):
        return input
    
    def backward(self, grad_wrt_output):
        raise NotImplementedError('backward')



class Linear(Layer):

    def __init__(self, number_nodes, input_dim=-1):
        self.output_dim = number_nodes
        self.input_dim = input_dim
        self.number_params = -1
        self.weights = None
        self.bias = None
        self.output = None
        self.input = None
        self.weights_grad = None
        self.bias_grad = None

# see if output needed
    def forward(self, input):
        self.input = input
        output = self.weights @ input + self.bias
        self.output = output
        return output
    
    def backward(self, *grad_wrt_output):
        # delta = layer.backward(deltas[-1]) * activation.backward(self.s_list[i])
        prev_delta = grad_wrt_output[0]
        activ_back = grad_wrt_output[1]
        curr_delta = (self.weights @ prev_delta) * activ_back #TODO : check if @ or transpose 
        self.weights_grad = curr_delta @ self.input
        self.bias_grad = curr_delta

        return 
    
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


    def __str__(self):
        return super().__str__() + ": Convolution 1D" 