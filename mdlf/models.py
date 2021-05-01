
from mdlf.loss import MSE
import mdlf.activations as activations
import mdlf.layers as layers

class Sequential:

    def __init__(self, *modules):
        self.optimizer = None
        self.loss = None
        self.modules = list(modules)
        self.number_params = 0
        self.s_list = []
        self.x_list = []

        self.layers_activations = []


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
        self.s_list = []

        tmp = input    
        
        self.x_list = [input]
        for module in self.modules:
            output = module.forward(tmp)
            if isinstance(module, layers):
                self.s_list.append(output)
            elif isinstance(module, activations):
                self.x_list.append(output)
            tmp = output

        return tmp
    
    def backward(self, input, label):
        #simple case : when begin with layer, alternate with activations / layers, and ends with an activation
        output = self.forward(input)

        last_s = self.s_list[-1]
        last_activation = self.layers_activations[-1][1]
        delta = self.loss.backward(self.forward(input), label) @ last_activation.backward(last_s)  #check if transpose needed or *
        deltas = [delta]

        grad_w = []
        for i, (layer, activation) in reversed(list(enumerate(self.layers_activations))):
            delta = layer.backward(deltas[-1]) * activation.backward(self.s_list[i])
            deltas.append(delta)

            if isinstance(layer, layers.Linear):
                grad_w_i = delta @ self.x_list[i - 1]
                grad_b_i = delta
                grad_w.append(())


        return None

    def train(self, train_data, train_label):
        return None

    def compile(self, optimizer, loss):

        if optimizer.upper() == 'SGD':
            None
            #TODO : determines if better to do this into another file (to not import MSE here)
        if loss.upper() == 'MSE':
            self.loss = MSE

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