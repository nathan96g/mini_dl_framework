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

model.add(activations.ReLU())
model.add(layers.Linear(number_nodes=25, input_dim=2))
model.add(activations.ReLU())
model.add(layers.Linear(number_nodes=25))
model.add(layers.Linear(number_nodes=25))
model.add(activations.ReLU())
model.add(activations.ReLU())
model.add(layers.Linear(number_nodes=25))
model.add(activations.ReLU())
model.add(layers.Linear(number_nodes=2))
model.add(activations.Tanh())

model.compile(optimizer=optimizer.SGD(), loss=loss.MSE())

print(model)

model.train(train_data, train_labels)

#print plot with or with not the predicted values form neural net 

"""
def plot_circle_with_predicted_labels (data, label, predicted_label=-1):
    x_cor, y_cor = data[:,0],data[:,1] 
    
    # if the function is given as input the predicted_label else otherwise 
    if type(predicted_label) != int:
        # -1 if correct value for 0, 1 if correct value for 1, 0 if uncorrect prediction 
        correctness = -label -predicted_label + 1 
        Accuracy = 1-correctness.tolist().count(0)/data.size(0) * 100
    else : 
        # -1 if correct value for 1, 1 if correct value for 0
        correctness= -2*label+1
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(x_cor[correctness==-1],y_cor[correctness==-1],'.' ,color='b')
    ax.plot(x_cor[correctness==1],y_cor[correctness==1],'.' ,color='r')
    ax.plot(x_cor[correctness==0],y_cor[correctness==0],'.' ,color='yellow')

    circle = plt.Circle((0.5, 0.5), 1 / ((2 * math.pi)**0.5), 
                        color='black', fill=False, linewidth=2.5)
    ax.set_facecolor('lightblue') 
    ax.add_artist(circle)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
        
    ax.set_xlabel('X axis', fontsize = 20)
    ax.set_ylabel('Y axis', fontsize = 20)  
    ax.tick_params(axis="x",labelsize = 15)
    ax.tick_params(axis="y",labelsize = 15)
    
    if type(predicted_label) != int:
            ax.set_title('Predicted values for test set \n  Accuracy = {}'.format(Accuracy), fontsize = 25, fontweight='bold',pad=20)
            fig.savefig('Predicted_values'+'.png')
    else : 
            ax.set_title('Training set', fontsize = 25, fontweight='bold',pad=20)
            fig.savefig('Training_set'+'.png')          
"""