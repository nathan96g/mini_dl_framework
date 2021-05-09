from mdlf import loss, optimizer
from torch import empty
import math
import mdlf.models as models
import mdlf.layers as layers
import mdlf.activations as activations

#to be deleted when submission
import utils

def generate_dataset(size):
    """
    Generates a training and a test set of 'size' points sampled uniformly in [0,1]^2, 
    each with a label 0 if outside the disk centered at (0.5,0.5) of radius 1/√2π, and 1 inside.
    """
    data = empty(size, 2).uniform_(0, 1)
    labels = data.sub(0.5).pow(2).sum(1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()
    return data, labels

train_data, train_labels = generate_dataset(1000)
test_data,  test_labels  = generate_dataset(1000)

# Create sequential model
model = models.Sequential()

model.add(layers.Linear(number_nodes=25, input_dim=2))
model.add(activations.ReLU())
model.add(layers.Linear(number_nodes=25))
model.add(activations.ReLU())
model.add(layers.Linear(number_nodes=25))
model.add(activations.ReLU())
model.add(layers.Linear(number_nodes=1))
model.add(activations.ReLU())

model.compile(optimizer=optimizer.SGD(), loss=loss.MSE())


loss,accuracy = model.train(train_data, train_labels)

for i in range(10):
    print("epochs number {} : Loss = {} and Accuracy = {}".format( i+1,loss[i] , accuracy[i] ))


#Apply tensorflow neural network on data
"""
utils.call_NN_tensorflow( train_data, train_labels, test_data, test_labels,
                epochs= 10, 
                show_accuracy = True,
                show_points = True)
"""