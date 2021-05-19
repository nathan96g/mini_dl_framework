from mdlf import loss, optimizer
from torch import empty
import math
import mdlf.models as models
import mdlf.layers as layers
import mdlf.activations as activations
import mdlf.metrics as metrics
import math

#to be deleted when submission
import utils
import torch
torch.manual_seed(42)
#print(torch.seed())

def generate_dataset(size):
    """
    Generates a training and a test set of 'size' points sampled uniformly in [0,1]^2, 
    each with a label 0 if outside the disk centered at (0.5,0.5) of radius 1/√2π, and 1 inside.
    """
    data = empty(size, 2).uniform_(0, 1)
    labels = data.sub(0.5).pow(2).sum(1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()
    return data, labels

train_data, train_label = generate_dataset(1000)
test_data,  test_label  = generate_dataset(1000)

# Create sequential model
model = models.Sequential()

model.add(layers.Linear(number_nodes=25, input_dim=2))
model.add(activations.ReLU())
model.add(layers.Linear(number_nodes=25))
model.add(activations.ReLU())
model.add(layers.Linear(number_nodes=25))
model.add(activations.ReLU())
model.add(layers.Linear(number_nodes=1))
model.add(activations.Sigmoid())

model.compile(optimizer=optimizer.SGD(lambda_ = 0.01), 
              loss=loss.MSE(), 
              metrics=metrics.BinaryAccuracy(threshold=0.5))

print(model)

#two way to call function train -> input only train or train + test
train_loss_per_epochs, train_accuracy_per_epochs = model.train(train_data, train_label, epochs=10)
train_loss_per_epochs, train_accuracy_per_epochs, test_loss_per_epochs, test_accuracy_per_epochs = model.train(train_data, train_label,epochs = 10, test_data =test_data, test_label= test_label)

#call function 
predicted_labels, test_loss,test_accuracy = model(test_data,test_label)

#Plot result of Mini DL framework
#utils.plot_circle_with_predicted_labels(test_data, test_label, predicted_label=predicted_labels)


#Apply tensorflow neural network on data
m = utils.call_NN_tensorflow( train_data, train_label, test_data, test_label,
                epochs= 10, 
                show_accuracy = True,
                show_points = True)
