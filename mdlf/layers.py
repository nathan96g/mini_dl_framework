import torch

class Linear:

    def __init__(self, number_nodes, input_dim=-1):
        self.output_dim = number_nodes
        self.input_dim = input_dim
        self.weights = None
        self.bias = None

    def forward(self, input):
        raise NotImplementedError('forward')
    
    def backward(self, grad_from_output):
        raise NotImplementedError('backward')
    
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

        # Initialize weights and bias
        self.weights = torch.empty(self.input_dim, self.output_dim).uniform_(0, 1)
        self.bias = torch.empty(self.output_dim).fill_(0)
    
    def __str__(self):
        return "Layer: Linear"