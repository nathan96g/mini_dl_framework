
class Module(object):
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
