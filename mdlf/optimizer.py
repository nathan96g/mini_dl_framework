from torch import empty

class Optimizer:
    def full_step(self, model, train_data, train_label):
        return NotImplementedError("full_step")

    def __str__(self):
        return "Optimizer"


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with a minibatch size
    fixed to 1.
    """
    def __init__(self, lambda_ = 0.01):
        self.lambda_ = lambda_

    def full_step(self, model, train_data, train_label):
        """
        Train a model for a full epoch by iterating over all the train dataset.
        """
        #when step is called, need to iterate on all the data points for 1 epoch
        random_int = empty(train_data.size(0)).uniform_(0, 1).sort().indices
        for n in random_int :
            train_sample= train_data[n]
            train_sample_label= train_label[n]

            # Set gradients to zero 
            model.grad_zero()
            # Computes gradients
            model.backward(train_sample, train_sample_label)

            # Updates the gradients
            for module in model.modules :
                updates = []
                for param in module.param():
                       weight = param[0]
                       weight_grad = param[1]
                       updates.append(weight - self.lambda_ * weight_grad)
                module.update(updates)
                
        return model

    def __str__(self):
        return super().__str__() + ": SGD"

#TODO : see if time to implement => require to store list of grad in modules and not just grad
class Minibatch_SGD(Optimizer):
    def __init__(self, lambda_, mini_batch_size):
        self.lambda_ = lambda_
        self.mini_batch_size = mini_batch_size

    def step(self, model, train_data, train_label):
        raise NotImplementedError