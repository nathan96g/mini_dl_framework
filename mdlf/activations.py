class Activations:
    
    def __init__(self, dim=-1):
        self.input_dim = dim
        self.output_dim = dim

    def forward(self, input):
        raise NotImplementedError('forward')
    
    def backward(self, grad_from_output):
        raise NotImplementedError('backward')
    
    def initialize(self, input_dim):
        self.input_dim = input_dim
        self.output_dim = input_dim

class ReLU(Activations):
   
    def forward(self, input):
        return None
    
    def backward(self, grad_from_output):
        return None
    
    def __str__(self):
        return "Activation: ReLU"

class Tanh(Activations):

    def forward(self, input):
        return None
    
    def backward(self, grad_from_output):
        return None
    
    def __str__(self):
        return "Activation: Tanh"