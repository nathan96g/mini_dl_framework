from torch import empty
import mdlf.activations as activations
import mdlf.layers as layers

class Optimizer:
    def step(self, model, train_data, train_label):
        return NotImplementedError

#TODO: determine get a good lambda_
class SGD(Optimizer):
    def __init__(self, lambda_ = 0.01):
        self.lambda_ = lambda_

    def step(self, model, train_data, train_label):
        #when step is called, need to iterate on all the data points for 1 epoch
        # random_int = empty(train_data.size(0)).uniform_(0, 1).sort().indices
        # print(random_int)
        # print(train_data.size(0))
        random_int = range(train_data.size(0))
        # random_int = range(train_data.size(0))
        for n in random_int :
            train_sample= train_data[n]
            train_sample_label= train_label[n]
            # print(train_sample_label.argmax())
            
            # set gradient to zero 
            model.grad_zero()
            
            model.backward(train_sample, train_sample_label)
            for module in model.modules :
                updates = []
                for param in module.param():
                       weight = param[0]
                    #    if weight.shape[0] == 8:
                    #        print("icic")
                    #        print(weight)
                       weight_grad = param[1]
                       weight_updated = weight - self.lambda_ * weight_grad
                       updates.append(weight_updated)
                module.update(updates)
        return model

#TODO : see if time to implement => require to store list of grad in modules and not just grad
class Minibatch_SGD(Optimizer):
    def __init__(self, lambda_, mini_batch_size):
        self.lambda_ = lambda_
        self.mini_batch_size = mini_batch_size

    def step(self, model, train_data, train_label):
        raise NotImplementedError