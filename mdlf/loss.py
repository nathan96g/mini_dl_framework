from mdlf.module import Module


class Loss(Module):   
    #TODO : just evaluation : return it and print it
    def forward(self, input, label): 
        raise NotImplementedError
    #TODO : just derivate
    def backward(self, input,label): 
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class MSE(Loss):

    def __init__(self):
        self.input = None
        self.label = None

    #TODO
    def forward(self, input, label): 
        self.input = input
        self.label = label
        loss_mse = (0.5*(input - label) ** 2)
        return loss_mse
        #raise NotImplementedError

    #TODO : change it
    def backward(self, input, label):
        outpout =  input - label
        return outpout
        #return gradwrtoutput[0]

    #TODO : See if needed
    def __str__(self):
        raise NotImplementedError