
class Module(object):
    def forward(self, *input): 
        raise NotImplementedError

    def backward(self, *gradwrtoutput): 
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError