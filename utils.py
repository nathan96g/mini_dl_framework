################################################
import math
from torch import empty
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from tensorflow.keras import layers as layers_tf
from tensorflow.keras import models as models_tf

# print plot with or with not the predicted values form neural net 
def plot_circle_with_predicted_labels (data, label, predicted_label=-1, tensorflow = False):
    x_cor, y_cor = data[:,0],data[:,1] 
    
    # if the function is given as input the predicted_label else otherwise 
    if type(predicted_label) != int:
        # -1 if correct value for 0, 1 if correct value for 1, 0 if uncorrect prediction 
        correctness = -label - torch.tensor(predicted_label) + 1 
        Accuracy = 100 - round(correctness.tolist().count(0)/data.size(0) * 100, 2)
        
    else : 
        # -1 if correct value for 1, 1 if correct value for 0
        correctness= -2*label+1
    
    fig, ax = plt.subplots(figsize=(8,8))
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
    
    if type(predicted_label) != int and tensorflow == True :
            ax.set_title('TensorFlow \n Predicted values for test set \n  Accuracy = {}'.format(Accuracy), fontsize = 20, fontweight='bold',pad=5)
            fig.savefig('Figure/Predicted_values_tensorflow'+'.png')
    elif type(predicted_label) != int and tensorflow == False :
            ax.set_title('Mini DL framework \n Predicted values for test set \n  Accuracy = {}'.format(Accuracy), fontsize = 20, fontweight='bold',pad=5)
            fig.savefig('Figure/Predicted_values_framework'+'.png')
    else : 
            ax.set_title('Training set', fontsize = 25, fontweight='bold',pad=20)
            fig.savefig('Figure/Training_set'+'.png')  

################################################ 

def call_NN_tensorflow(train_data,train_labels,test_data,test_labels,
                    epochs = 10,
                    show_accuracy = False,
                    show_points = False):
                            
    model_tf = models_tf.Sequential()
    model_tf.add(tf.keras.Input(shape=(train_data.size(1),)))
    model_tf.add(layers_tf.Dense(25, activation='relu'))
    model_tf.add(layers_tf.Dense(25, activation='relu'))
    model_tf.add(layers_tf.Dense(25, activation='relu'))
    model_tf.add(layers_tf.Dense(1, activation='tanh'))
    #model_tf.add(layers_tf.Dense(1, activation="sigmoid"))

    model_tf.summary()

    metrics = tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.0)
    metrics2 = tf.keras.metrics.BinaryAccuracy(name="accuracy")

    model_tf.compile(optimizer='SGD',loss='MSE',metrics=[metrics, metrics2])

    history = model_tf.fit(train_data.tolist(), 
                           train_labels.tolist(), 
                           epochs=epochs,
                           validation_data=(test_data.tolist(), test_labels.tolist()),
                           batch_size=1)

    if show_accuracy :
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'test_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right') 
        plt.savefig('Figure/Accuracy'+'.png')

    if show_points :
        predictions = model_tf.predict(test_data.tolist())
        predictions =[1 if i >=0.0 else 0 for i in predictions]
        plot_circle_with_predicted_labels(test_data,test_labels,predictions,tensorflow = True)



################################################

def plot_result(label, predictions_per_epochs):

    #création de epoch_step qui est une liste des epochs qu'on va afficher 
    length = len(predictions_per_epochs)
    epoch_step = []
    step = round(length/5)
    for i in range(0, length, step):
        epoch_step.append(i)
    epoch_step.append(length-1)
    
    #création des trois columns 
    #list de tous les labels 
    final_label = []
    #list de tous les données à la suite 
    final_data = []
    #list du nombre d'epochs 
    final_epochs = []

    # concatenation des résultats des epochs intéréssés avec predictions_per_epochs[i]
    for i in epoch_step : 
        final_data= final_data + predictions_per_epochs[i].tolist() 
        final_label = final_label + label.tolist()
        #indication de quel epochs on est en train de parler 
        final_epochs = final_epochs + [i+1] * len(label) 
    
    
    #création d'un data frame avec trois columns pour etre utiliser par striplot 
    df = pd.DataFrame(list(zip(final_data, final_label)),
               columns =['Prediction from the NN', 'label'])

    df['number of epochs'] = final_epochs

    sns.stripplot(y='Prediction from the NN', x='number of epochs', 
                   data=df, 
                   jitter=True,
                   dodge=True,
                   marker='o', 
                   alpha=0.5,
                   hue='label')

    plt.savefig('Figure/plot_prediction_per_epochs.png')
    