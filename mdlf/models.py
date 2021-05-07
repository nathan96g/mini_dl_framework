
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

#TODO : Doesn't work, used to instantiate layers_activations and correct model: 
# will correct the model to get always an alternance (layer, activation):
# will add activation identity of identity layer.
# begins with layer, ends with activation 
    def reconstruct_model(self):
        module_list = []
        prev_is_layer = False
        for module in self.modules:
            if issubclass(type(module),activations.Activation):
                module_list.append(module)
                if not prev_is_layer:
                    module_list.append(layers.Identity())
            elif issubclass(type(module), layers.Layer):
                module_list.append(module)
                if prev_is_layer:
                    module_list.append(activations.Identity())
                prev_is_layer = True

        #TODO: if doesn't finish with activation add identity layer
        self.modules = module_list
        return True

#TODO : add loss to modules at the end
#modify subsequently backward by suppressing the last_activation and init delta to None
#modify also loss backward to get only 1 possible case

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
        #TODO: change it after corrected reconstruct model or transform layer form to get directly (layer, activation) together
        # self.reconstruct_model()
        for e in range(epochs) :
            self.optimizer.step(self, train_data, train_label)
        return self

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

        previous_output_dim = -1
        for module in self.modules:
            module.initialize(input_dim = previous_output_dim)
            previous_output_dim = module.output_dim
            self.number_params += module.number_params
        return
    
    def __str__(self):

        descriptions = [[str(module), str(module.input_dim), str(module.output_dim), str(module.number_params)] for module in self.modules]
        descriptions = [['Modules', 'Input dimension', 'Output dimension', 'Number of parameters']] + descriptions

        lens = [max(map(len, col)) for col in zip(*descriptions)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in descriptions]

        return "\n\nSequential model, {} modules, {} total number of parameters: \n \n".format(len(self.modules), self.number_params)+'\n'.join(table)