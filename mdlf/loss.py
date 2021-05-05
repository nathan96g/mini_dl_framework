from mdlf.module import Module


class Loss(Module):   
    #just evaluation
    def forward(self, *input): 
        raise NotImplementedError
    #just derivate
    def backward(self, *gradwrtoutput): 
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class MSE(Loss):
    def forward(self, *input): 
        raise NotImplementedError

    def backward(self, *gradwrtoutput): 
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError