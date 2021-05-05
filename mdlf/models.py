
import mdlf.activations as activations
import mdlf.layers as layers

class Sequential:

    def __init__(self, *modules):
        self.optimizer = None
        self.loss = None
        self.modules = list(modules)
        self.number_params = 0

        self.layers_activations = []
        self.optimizer = None


    def add(self, module):
        self.modules.append(module)

#TODO : Check if right implementation, right way to do things
    def reconstruct_model(self):
        layers_list = []
        activations_list = []
        prev_is_layer = False
        for module in self.modules:
            if isinstance(module, activations):
                activations_list.append(module)
                if not prev_is_layer:
                    layers_list.append(layers.Identity())
            elif isinstance(module, layers):
                layers_list.append(module)
                if prev_is_layer:
                    activations_list.append(activations.Identity())
                prev_is_layer = True

        #doesn't finish with activation 
        if len(layers_list) > len(activations_list):
            activations_list.append(activations.Identity())

        if len(layers_list) != len(activations_list):
            print("error implementation reconstruct_model")
            raise Exception
        
        self.layers_activations = zip(layers_list, activations_list)  
        return self.layers_activations      


    def forward(self, input):
        tmp = input    
        for module in self.modules:
            output = module.forward(tmp)
            tmp = output
        return tmp
    
    def backward(self, input, label):
        #simple case : when begin with layer, alternate with activations / layers, and ends with an activation
        output = self.forward(input)

        last_activation = self.layers_activations[-1][1]
        delta = self.loss.backward(self.forward(input), label) @ last_activation.backward(output)  #check if transpose needed or *

        for layer, activation in reversed(self.layers_activations):
            delta = layer.backward(delta, activation.backward())

        return delta #useless normally

    def train(self, train_data, train_label, epochs = 1):
        for e in epochs :
            self.optimizer(self, train_data, train_label)
        return None

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