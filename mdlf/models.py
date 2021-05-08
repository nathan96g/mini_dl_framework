import mdlf.activations as activations
import mdlf.layers as layers

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

    def forward(self, input):
        tmp = input    
        
        for module in self.modules:
            output = module.forward(tmp)
            tmp = output
        return tmp
    
        #simple case : when begin with layer, alternate with activations / layers, and ends with an activation
        #TODO : implement reconstruct to modify modules
    def backward(self, input, label):
        output = self.forward(input)
        last_activation = self.modules[-1]

        delta = self.loss.backward(output, label) * last_activation.backward()  #check if transpose needed or @
        for module in reversed(self.modules[:-1]): #get last layer (since -1 )
            delta = module.backward(delta)

        return delta #useless normally

    def train(self, train_data, train_label, epochs = 10):
        for e in range(epochs) :
            self.optimizer.step(self, train_data, train_label)
        return self

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
        self.reconstruct_model()

        previous_output_dim = -1
        for module in self.modules:
            module.initialize(input_dim = previous_output_dim)
            previous_output_dim = module.output_dim
            self.number_params += module.number_params
        return
    
    def __str__(self):

        #does not show Identity Activation or Identity Layer 
        descriptions = [[str(module), str(module.input_dim), str(module.output_dim), str(module.number_params)] for module in self.modules if not (issubclass(type(module),activations.Identity) or issubclass(type(module),layers.Identity)) ]
        descriptions = [['Modules', 'Input dimension', 'Output dimension', 'Number of parameters']] + descriptions

        lens = [max(map(len, col)) for col in zip(*descriptions)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in descriptions]

        return "\n\nSequential model, {} modules, {} total number of parameters: \n \n".format(len(self.modules), self.number_params)+'\n'.join(table)