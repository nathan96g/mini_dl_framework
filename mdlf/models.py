from mdlf.metrics import Accuracy
import torch
import mdlf.activations as activations
import mdlf.layers as layers
from torch import empty

class Sequential:
    """
    A sequential model allow the concatenation of
    modules in a sequential manners, i.e. the first module
    transforms this input, forward the ouput to the next module
    , etc.
    """

    ########################################
    #           INITIALIZE MODEL           #
    ########################################

    def __init__(self, *modules):
        """
        To initialize a model you can do either:
            
            model = Sequential(module0, module1, ...)

        or

            model = Sequential()
            model.add(module0)
            model.add(module1)
            ...
        """
        self.optimizer = None
        self.loss = None
        self.modules = list(modules)
        self.number_params = 0

        self.optimizer = None
        self.metrics = None

        self.param_test = []

    
    def add(self, module):
        """
        Need to be called before the compile routine.
        """
        self.modules.append(module)

    ########################################
    #             COMPILE MODEL            #
    ########################################

    def reconstruct_model(self):
        """
        Make sure that the model begins with layer and 
        ends with on activation.
        This function also make sure that the alternance
        “layer-activation“ is respected.
        This function is called in the compile routine.
        """
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
        return

    def compile(self, optimizer, loss, metrics=None):
        """
        Add optimizer, loss and metrics to the model,
        call the reconstruct_routine, and initialize
        correctly the input and output dimension of 
        each sub-module.
        """
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
        return

    ########################################
    #        FORWARD & BACKWARD PASS       #
    ########################################

    def forward(self,input):
        """
        Forward the input through the model
        and return the prediction.
        """
        tmp = input
        
        for module in self.modules:
            output = module.forward(tmp)
            tmp = output

        return output
    

    def backward(self, input, label):
        """
        Compute the back-propagation over the model
        """
        output = self.forward(input)

        delta = self.loss.backward(output, label)
        for module in reversed(self.modules):
            delta = module.backward(delta)

        return delta #useless normally

    ########################################
    #               TRAINING               #
    ########################################

    def grad_zero(self):
        """
        Set all the gradient of all modules to zero
        """
        for module in self.modules:
            if issubclass(type(module),layers.Linear):
                module.gradient_to_zero()


    def train(self, train_data, train_label, epochs = 10, test_data = empty((0,0))  , test_label = empty((0,0)), verbal=True):
        """
        Use the optimizer given at the compilation to train this model.
        If test_data and test_label are given, we compute the loss and 
        the accuracy for the test set at the end of each epoch.
        If verbal is true, we print the loss and accuracy at the end of
        each epoch.
        """
        losses_train = []      # per epoch
        accuracies_train  = [] # per epoch
        predictions = []       # per epoch
        test_is_here = False

        # Check if the test is given
        if test_data.size() != (0,0) and test_label.size() != (0,0) :  
            test_is_here =  True 
            losses_test = []      # per epoch
            accuracies_test  = [] # per epoch

        for i in range(epochs) :
            self.optimizer.full_step(self, train_data, train_label)
            pred, loss,accuracy = self(train_data,train_label)
            losses_train.append(loss)
            accuracies_train.append(accuracy)
            predictions.append(pred)

            if test_is_here :
                _, loss,accuracy = self(test_data,test_label)
                losses_test.append(loss)
                accuracies_test.append(accuracy)
            
            # Print accuracy and loss per epoch
            if verbal :
                if test_is_here:
                    print("epochs number {} : train loss = {}, train accuracy = {}, test loss = {}, test accuracy = {}"
                                .format( i+1,losses_train[-1] , accuracies_train[-1], losses_test[-1] , accuracies_test[-1]))
                else:
                    print("epochs number {} : train loss = {}, train accuracy = {}"
                                .format( i+1,losses_train[-1] , accuracies_train[-1]))
        
        # Return all predictions, losses and accuracies per epoch for statistics purpous.
        if test_is_here : 
            return predictions, losses_train, accuracies_train, losses_test, accuracies_test
        else : 
            return predictions, losses_train, accuracies_train


    def __call__(self, data, label=None):
        """
        If label is None, only return the prediction for data.
        Return the loss and the accuracy for the label predicted by
        this model (self) w.r.t. the associated labels
        """
        # If we only want predictions
        if label == None:

            size_ = data.shape
            # Check if we give a dataset or a single sample
            if self.modules[0].input_dim == size_:
                return self.forward(data)

            output = torch.empty(size_[0])
            for i in range(size_[0]):
                output[i] = self.forward(data[i])
            return output

        size_ = label.shape

        output = torch.empty(size_)

        # Handle the case where data is only one sample
        if size_[0] == 1:
            data = data.view(1,-1)
        
        for i in range(size_[0]):
            output[i] = self.forward(data[i])

        accuracy = self.metrics(output, label)
        return output, self.loss.forward(output,label), accuracy
    
    def __str__(self):
        """
        Return a string to pretty-print this model
        """
        #Does not show Identity Activation or Identity Layer 
        descriptions = [[str(module), str(module.input_dim), str(module.output_dim), str(module.number_params)] for module in self.modules if not (issubclass(type(module),activations.Identity) or issubclass(type(module),layers.Identity)) ]
        descriptions = [['Modules', 'Input dimension', 'Output dimension', 'Number of parameters']] + descriptions

        lens = [max(map(len, col)) for col in zip(*descriptions)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in descriptions]

        return "\n\nSequential model, {} modules, {} total number of parameters: \n \n".format(len(self.modules), self.number_params)+'\n'.join(table)