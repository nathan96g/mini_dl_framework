from mdlf.metrics import Accuracy
import torch
import mdlf.activations as activations
import mdlf.layers as layers
from torch import empty

class Sequential:

    def __init__(self, *modules):
        self.optimizer = None
        self.loss = None
        self.modules = list(modules)
        self.number_params = 0

        self.layers_activations = None
        self.optimizer = None
        self.metrics = None

        self.param_test = []

    
    def add(self, module):
        self.modules.append(module)


##### Mise a jour #####
#  begins with layer and ends with activation 
# + correct model if alternance “layer-activation“ is not respected
    def reconstruct_model(self):
        module_list = []
        prev_is_layer = False

        #check if first module is activation, if yes add a layer identity before
        if issubclass(type(self.modules[0]),activations.Activation): 
            module_list.append(layers.Identity())
            prev_is_layer = True
        
        for module in self.modules:
            if prev_is_layer :
                if issubclass(type(module),activations.Activation) :
                    module_list.append(module)
                    prev_is_layer = False  
                else : 
                    module_list.append(activations.Identity())
                    module_list.append(module)
                    prev_is_layer = True  
            else : 
                if issubclass(type(module),layers.Layer) :
                    module_list.append(module)  
                    prev_is_layer = True 
                else : 
                    module_list.append(layers.Identity())
                    module_list.append(module)
                    prev_is_layer = False 

        #check if last module is an activation, if no add activation identity 
        if prev_is_layer : 
            module_list.append(activations.Identity())

        self.modules = module_list
        return True


#TODO : add loss to modules at the end
#modify subsequently backward by suppressing the last_activation and init delta to None
#modify also activation backward to get only 1 possible case

    def forward(self,input):
        tmp = input
        
        for module in self.modules:
            output = module.forward(tmp)
            tmp = output

        return output
    
        #simple case : when begin with layer, alternate with activations / layers, and ends with an activation
        #TODO : implement reconstruct to modify modules
    def backward(self, input, label):
        output = self.forward(input)
        # print("back2")
        delta = self.loss.backward(output, label)
        for module in reversed(self.modules):
            # print(delta)
            delta = module.backward(delta)
        # print("backend2")
        return delta #useless normally


    def train(self, train_data, train_label, epochs = 10, test_data = empty((0,0))  , test_label = empty((0,0)), verbal=True):

        loss_per_epoch_train = []
        accuracy_per_epoch_train  = []
        test_is_here = False

        if test_data.size() != (0,0) and test_label.size() != (0,0) :  
            test_is_here =  True 
            loss_per_epoch_test = []
            accuracy_per_epoch_test  = []

        for i in range(epochs) :
            self.optimizer.step(self, train_data, train_label)
            loss,accuracy,_ = self.loss_accuracy_function(train_data,train_label)
            loss_per_epoch_train.append(loss)
            accuracy_per_epoch_train.append(accuracy)

            if test_is_here :
                loss,accuracy,_ = self.loss_accuracy_function(test_data,test_label)
                loss_per_epoch_test.append(loss)
                accuracy_per_epoch_test.append(accuracy)
            
            if verbal:
                print("epochs number {} : Loss = {} and Accuracy = {}".format( i+1,loss ,accuracy ))

        if test_is_here : return loss_per_epoch_train, accuracy_per_epoch_train, loss_per_epoch_test, accuracy_per_epoch_test
        else : return loss_per_epoch_train, accuracy_per_epoch_train 


    def loss_accuracy_function(self,train_data,train_label):
        size_ = train_label.shape
        output = torch.empty(size_)
        loss = torch.empty(size_)
        for i in range(size_[0]):
            output[i] = self.forward(train_data[i])
            loss[i] = self.loss.forward(output[i],train_label[i])

        accuracy = self.metrics(output, train_label)
        return loss.sum(), accuracy, output
    
    def __call__(self,test_data,test_label):
        test_loss,test_accuracy, predicted_labels = self.loss_accuracy_function(test_data,test_label)
        return test_loss,test_accuracy,predicted_labels

    def compile(self, optimizer, loss, metrics=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.reconstruct_model()

        previous_output_dim = -1
        for module in self.modules:
            module.initialize(input_dim = previous_output_dim)
            if issubclass(type(module),layers.Layer) :
                par = module.param()
                self.param_test.append((torch.clone(par[0][0]), torch.clone(par[1][0])))
            previous_output_dim = module.output_dim
            self.number_params += module.number_params
        
        #add loss module at the end of modules 
        #self.modules.append(loss)

        return
    
    def grad_zero(self):
        for module in self.modules:
            if issubclass(type(module),layers.Linear):
                module.gradient_to_zero()
    
    def __str__(self):

        #does not show Identity Activation or Identity Layer 
        descriptions = [[str(module), str(module.input_dim), str(module.output_dim), str(module.number_params)] for module in self.modules if not (issubclass(type(module),activations.Identity) or issubclass(type(module),layers.Identity)) ]
        descriptions = [['Modules', 'Input dimension', 'Output dimension', 'Number of parameters']] + descriptions

        lens = [max(map(len, col)) for col in zip(*descriptions)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in descriptions]

        return "\n\nSequential model, {} modules, {} total number of parameters: \n \n".format(len(self.modules), self.number_params)+'\n'.join(table)