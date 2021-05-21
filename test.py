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
model.add(activations.Tanh())

model.compile(optimizer=optimizer.SGD(), 
              loss=loss.MSE(), 
              metrics=metrics.BinaryAccuracy(threshold=0.5))

print(model)

#two way to call function train -> input only train or train + test
train_loss_per_epochs, train_accuracy_per_epochs,predicted_labels_per_epoch_train = model.train(train_data, train_label, epochs=5)
#train_loss_per_epochs, train_accuracy_per_epochs, test_loss_per_epochs, test_accuracy_per_epochs = model.train(train_data, train_label,epochs = 10, test_data =test_data, test_label= test_label)


#fit function 
#test_loss,test_accuracy,predicted_labels = model(test_data,test_label)

#Plot result of Mini DL framework
#utils.plot_circle_with_predicted_labels(test_data, test_label, predicted_label=predicted_labels)


#utils.plot_result(train_label, predicted_labels_per_epoch_train)


#Apply tensorflow neural network on data
"""
m = utils.call_NN_tensorflow( train_data, train_label, test_data, test_label,
                epochs= 10, 
                show_accuracy = True,
                show_points = True)
"""
epochs = 50
y_hat_list = []
for i in range(epochs) :
    y_hat = utils.call_NN_tensorflow( train_data, train_label, test_data, test_label,
                epochs= i+1, 
                show_accuracy = True,
                show_points = True)
    y_hat_list.append(torch.squeeze(torch.tensor(y_hat)))
    


"""
print(type(predicted_labels_per_epoch_train))
print(type(predicted_labels_per_epoch_train[0]))
print(type(predicted_labels_per_epoch_train[0][0]))
print(len(predicted_labels_per_epoch_train))
print(len(predicted_labels_per_epoch_train[0]))
#print(len(predicted_labels_per_epoch_train[0][0]))

print(type(m))
print(type(m[0]))
print(type(m[0][0]))
print(len(m))
print(len(m[0]))
#print(len(m[0][0]))

print(type(train_label))
print(type(test_label))

print(len(train_label))
print(len(test_label))
print(predicted_labels_per_epoch_train)
print(m)
"""
#print(predicted_labels_per_epoch_train)
#print(m)
utils.plot_result(test_label, y_hat_list)

"""
"""