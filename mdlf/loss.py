from mdlf.module import Module


class Loss(Module):   
    #TODO : just evaluation
    def forward(self, *input): 
        raise NotImplementedError
    #TODO : just derivate
    def backward(self, *gradwrtoutput): 
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class MSE(Loss):
    #TODO
    def forward(self, *input): 
        raise NotImplementedError
    #TODO
    def backward(self, *gradwrtoutput): 
        raise NotImplementedError
    #TODO : See if needed
    def __str__(self):
        raise NotImplementedError