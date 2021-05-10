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

    def forward(self, input, label):
        tmp = input    
        
        for module in self.modules:
            output = module.forward(tmp)
            tmp = output

        outpout = self.loss.forward(tmp,label)

        return outpout
    
        #simple case : when begin with layer, alternate with activations / layers, and ends with an activation
        #TODO : implement reconstruct to modify modules
    def backward(self, input, label):
        output = self.forward(input,label)
        last_activation = self.modules[-1]


        delta = self.loss.backward(output, label) * last_activation.backward()  #check if transpose needed or @
        for module in reversed(self.modules[:-1]): #get the second last layer (since -1 )
            delta = module.backward(delta)

        return delta #useless normally

    def train(self, train_data, train_label, epochs = 10, test_data = empty((0,0))  , test_label = empty((0,0))):

        loss_per_epoch_train = []
        accuracy_per_epoch_train  = []
        test_is_here = False

        if test_data.size() != (0,0) and test_label.size() != (0,0) :  
            test_is_here =  True 
            loss_per_epoch_test = []
            accuracy_per_epoch_test  = []

        for e in range(epochs) :
            self.optimizer.step(self, train_data, train_label)
            loss,accuracy,_ = self.loss_accuracy_function(train_data,train_label)
            loss_per_epoch_train.append(loss)
            accuracy_per_epoch_train.append(accuracy)

            if test_is_here :
                loss,accuracy,_ = self.loss_accuracy_function(test_data,test_label)
                loss_per_epoch_test.append(loss)
                accuracy_per_epoch_test.append(accuracy)

        if test_is_here : return loss_per_epoch_train, accuracy_per_epoch_train, loss_per_epoch_test, accuracy_per_epoch_test
        else : return loss_per_epoch_train, accuracy_per_epoch_train 

    def loss_accuracy_function(self,train_data,train_label):
        loss = []
        prediction = []
        size = train_data.size(0)

        for i in range (size):
            loss.append(self.forward(train_data[i],train_label[i]))
            prediction.append(self.loss.input) 

        prediction =[1 if n >=0.5 else 0 for n in prediction]
        result = [i1 - i2 for i1, i2 in zip(prediction, train_label.tolist())]
        accuracy = 100 - result.count(0)/size * 100

        return sum(loss), accuracy, prediction
    
    def fit(self,test_data,test_label):
        test_loss,test_accuracy, predicted_labels = self.loss_accuracy_function(test_data,test_label)
        return test_loss,test_accuracy,predicted_labels

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
        self.reconstruct_model()

        previous_output_dim = -1
        for module in self.modules:
            module.initialize(input_dim = previous_output_dim)
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