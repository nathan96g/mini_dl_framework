from mdlf.module import Module
from torch import empty
import torch
# import test2

class Layer(Module):
    def __str__(self):
        return "Layer"

    def param(self): 
        raise NotImplementedError('param')

    def update(self, new_weights):
        raise NotImplementedError('update')

class Identity(Layer):
    """
    This layer simply forward the input without
    any computation.
    """

    def __init__(self, input_dim=-1):
        self.output_dim = input_dim
        self.input_dim = input_dim
        self.number_params = 0

    def forward(self, input):
        return input
    
    def backward(self, grad_wrt_output):
        return grad_wrt_output

    
    def initialize(self, input_dim):
        """
        Correctly initialize the input and output dimension
        of this layer. Called when we compile the model.
        """

        # if the first module is an identity layer -> input_dim is size of train_data (2)
        # Handle automatic dimension initialization
        if input_dim == -1:
            if self.input_dim == -1:
                raise RuntimeError("This identity layer need a manual initialization "
                                   "of the input dimension, e.g."
                                   " Identity(input_dim=64)")
        else:
            if self.input_dim != -1 and input_dim != self.input_dim:
                raise RuntimeError("The manually set input dimension ({}) of this layer"
                                   " is different from the one inferred automatically ({})"
                                   " by the initialization of the sequential model".format(self.input_dim, input_dim))
            else:
                self.input_dim = input_dim
        
        self.output_dim = self.input_dim
    
    def param(self): 
        return []

    def update(self, *new_weights):
        return []


class Linear(Layer):
    """
    A fully connected layer, i.e. each node in the
    previous layer is connected to each node in this
    layer.
    """

    def __init__(self, number_nodes, input_dim=-1):
        self.output_dim = number_nodes
        self.input_dim = input_dim
        self.number_params = -1
        self.weights = None
        self.bias = None
        self.input = None
        self.weights_grad = None
        self.bias_grad = None


    
    
    def forward(self, input):
        # if we are in layer l:
        # input : x_{l-1} 
        # output : z_l
        self.input = input # we conserve this for the backward pass
        return self.weights @ input + self.bias
    
    def backward(self, *grad_wrt_output):
        curr_delta = grad_wrt_output[0] #delta_l
        self.weights_grad = curr_delta.view(-1, 1).mm(self.input.view(1, -1))
        self.bias_grad = curr_delta
        # delta_{l-1} without componentwise activation mult 
        return (self.weights.T @ curr_delta)


    def param(self): 
        return [(self.weights, self.weights_grad), (self.bias, self.bias_grad)]

    def update(self, new_weights):
        self.weights = new_weights[0]
        self.bias = new_weights[1]



    def gradient_to_zero(self):
        self.weights_grad[:, :] = 0.0
        self.bias_grad[:] = 0.0
    
    def initialize(self, input_dim):
        """
        Correctly initialize the input and output dimension
        of this layer. Called when we compile the model.
        """
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

        # Initialize weights and bias :
        uniform_param = 1 / (self.input_dim**(1/2))
        self.weights = empty(self.input_dim, self.output_dim).uniform_(-uniform_param, uniform_param).T 
        self.bias = empty(self.output_dim).fill_(0.0)
        self.number_params = self.output_dim * self.input_dim + self.output_dim
        
        # Initialize gradient weights and gradient bias :
        self.weights_grad = empty((self.weights.shape)).fill_(0.0)
        self.bias_grad = empty((self.bias.shape)).fill_(0.0)



    def __str__(self):
        return super().__str__() + ": Linear" 