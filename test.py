import torch
import math
import mdlf.sequential
import mdlf.linear
import mdlf.activations

def generate_dataset(size):
    """
    Generates a training and a test set of 'size' points sampled uniformly in [0,1]^2, 
    each with a label 0 if outside the disk centered at (0.5,0.5) of radius 1/√2π, and 1 inside.
    """
    data = torch.empty(size, 2).uniform_(0, 1)
    labels = data.sub(0.5).pow(2).sum(1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()
    return data, labels

d,l = generate_dataset(1000)

model = mdlf.sequential.Sequential()

model.add(mdlf.linear.Linear(number_nodes=25, input_dim=2))
model.add(mdlf.activations.ReLU())
model.add(mdlf.linear.Linear(number_nodes=25))
model.add(mdlf.activations.ReLU())
model.add(mdlf.linear.Linear(number_nodes=25))
model.add(mdlf.activations.ReLU())
model.add(mdlf.linear.Linear(number_nodes=1))
model.add(mdlf.activations.Tanh())

model.compile(optimizer='SGD', loss='MSE')

print(model)