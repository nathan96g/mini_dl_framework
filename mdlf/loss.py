from mdlf.module import Module


class Loss(Module):   
    #TODO : just evaluation : return it and print it
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
    #TODO : change it
    def backward(self, *gradwrtoutput): 
        return gradwrtoutput[0]
    #TODO : See if needed
    def __str__(self):
        raise NotImplementedError