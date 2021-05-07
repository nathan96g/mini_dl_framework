from mdlf import loss, optimizer
from torch import empty
import math
import mdlf.models as models
import mdlf.layers as layers
import mdlf.activations as activations

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
model.add(layers.Linear(number_nodes=2))
model.add(activations.Tanh())

model.compile(optimizer=optimizer.SGD, loss=loss.MSE)
print(model)

model.train(train_data, train_labels)
