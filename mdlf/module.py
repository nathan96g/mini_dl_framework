
class Module(object):
    """
    General module for almost every element
    of a deep learning model (layers, losses, activations)
    """

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput): 
        raise NotImplementedError
    
    def param(self): 
        return []

    def update(self, *new_weights):
        return []

    def __str__(self):
        raise NotImplementedError
