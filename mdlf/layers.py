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

    def __init__(self, input_dim=-1):
        self.output_dim = input_dim
        self.input_dim = input_dim
        self.number_params = 0

    def forward(self, input):
        return input
    
    def backward(self, grad_wrt_output):
        return grad_wrt_output
        #raise NotImplementedError('backward')
    
    def initialize(self, input_dim):
        #if the first module is an identity layer -> input_dim is size of train_data (2)
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
        return self.weights @ input + self.bias #!!! no .T
    
    def backward(self, *grad_wrt_output):
        #print("grad_wrt_output",grad_wrt_output)
        curr_delta = grad_wrt_output[0] #delta_l
        self.weights_grad = curr_delta.view(-1, 1).mm(self.input.view(1, -1))
        self.bias_grad = curr_delta     
        return (self.weights.T @ curr_delta) # delta_{l-1} without componentwise activation mult !!! .T


    def param(self): 
        return [(self.weights, self.weights_grad), (self.bias, self.bias_grad)]

    def update(self, new_weights):

        self.weights = new_weights[0]


        # print("new_bias",new_weights[1])
        self.bias = new_weights[1]


    def gradient_to_zero(self):
        self.weights_grad[:, :] = 0.0
        self.bias_grad[:] = 0.0
    
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

        # Initialize weights and bias :
        uniform_param = 1 / (self.input_dim**(1/2))
        self.weights = empty(self.input_dim, self.output_dim).uniform_(-uniform_param, uniform_param).T #!!! T
        # print(self.weights.shape)
        # print(self.weights[0: 3])
        # if self.input_dim == 5 :
        #     self.weights = torch.tensor([[ 0.3419,  0.3712, -0.1048,  0.4108, -0.0980],
        # [ 0.0902, -0.2177,  0.2626,  0.3942, -0.3281],
        # [ 0.3887,  0.0837,  0.3304,  0.0606,  0.2156],
        # [-0.0631,  0.3448,  0.0661, -0.2088,  0.1140],
        # [-0.2060, -0.0524, -0.1816,  0.2967, -0.3530],
        # [-0.2062, -0.1263, -0.2689,  0.0422, -0.4417],
        # [ 0.4039, -0.3799,  0.3453,  0.0744, -0.1452],
        # [ 0.2764,  0.0697,  0.3613,  0.0489, -0.1410]])
        # else : 
        #     self.weights = torch.tensor([[ 0.0950, -0.0959,  0.1488,  0.3157,  0.2044, -0.1546,  0.2041,  0.0633],
        # [ 0.1795, -0.2155, -0.3500, -0.1366, -0.2712,  0.2901,  0.1018,  0.1464],
        # [ 0.1118, -0.0062,  0.2767, -0.2512,  0.0223, -0.2413,  0.1090, -0.1218],
        # [ 0.1083, -0.0737,  0.2932, -0.2096, -0.2109, -0.2109,  0.3180,  0.1178],
        # [ 0.3402, -0.2918, -0.3507, -0.2766, -0.2378,  0.1432,  0.1266,  0.2938],
        # [-0.1826, -0.2410,  0.1876, -0.1429,  0.2146, -0.0839,  0.2022, -0.2747],
        # [-0.1784,  0.1078,  0.0747, -0.0901,  0.2107,  0.2403, -0.2564, -0.1888],
        # [ 0.3237, -0.1193, -0.1253, -0.3421, -0.2025,  0.0883, -0.0467, -0.2566],
        # [ 0.0083, -0.2415, -0.3000, -0.1947, -0.3094, -0.2251,  0.3534,  0.0668],
        # [ 0.1090, -0.3298, -0.2322, -0.1177,  0.0553, -0.3111, -0.1523, -0.2117]])
        # print(self.weights.shape)
        # print(self.weights)
        
        self.bias = empty(self.output_dim).fill_(0.0)
        self.number_params = self.output_dim * self.input_dim + self.output_dim
        
        # Initialize gradient weights and gradient bias :
        self.weights_grad = empty((self.weights.shape)).fill_(0.0)
        self.bias_grad = empty((self.bias.shape)).fill_(0.0)

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