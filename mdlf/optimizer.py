from torch import empty


class Optimizer:
    def step(self, model, train_data, train_label):
        return NotImplementedError

#TODO: determine get a good lambda_
class SGD(Optimizer):
    def __init__(self, lambda_ = 0.001):
        self.lambda_ = lambda_

    def step(self, model, train_data, train_label):
        
        #TODO: take random element : https://discuss.pytorch.org/t/efficiently-selecting-a-random-element-from-a-vector/37757
        train_sample = train_data[0]
        train_sample_label = train_label[0]
        model.backward(train_sample, train_sample_label)
        for module in model.modules:
            updates = []
            for param in module.param():
                weight = param[0]
                weight_grad = param[1]
                updates.append(weight + self.lambda_ * weight_grad)
            module.update(updates)
        return model

#TODO : see if time to implement => require to store list of grad in modules and not just grad
class minibatch_SGD(Optimizer):
    def __init__(self, lambda_, mini_batch_size):
        self.lambda_ = lambda_
        self.mini_batch_size = mini_batch_size

    def step(self, model, train_data, train_label):
        raise NotImplementedError