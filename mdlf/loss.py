from mdlf.module import Module


class Loss(Module):   

    def forward(self, input, label): 
        raise NotImplementedError

    def backward(self, input,label): 
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class MSE(Loss):

    def __init__(self):
        self.input = None
        self.label = None

    def forward(self, input, label): 
        self.input = input
        self.label = label
        loss_mse = ((input - label) ** 2).mean()
        return loss_mse

    def backward(self, input, label):
        output =  2 * (input - label)
        return output


    def __str__(self):
        raise NotImplementedError