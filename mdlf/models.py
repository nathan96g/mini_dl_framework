class Sequential:

    def __init__(self, *modules):
        self.optimizer = None
        self.loss = None
        self.modules = list(modules)
        self.number_params = 0

    def add(self, module):
        self.modules.append(module)

    def forward(self, input):

        tmp = input

        for module in self.modules :
            output = module.forward(tmp)
            tmp = output

        return tmp
    
    def backward(self, grad_from_output):
        return None

    def train(self, train_data, train_label):
        return None

    def compile(self, optimizer, loss):

        if optimizer.upper() == 'SGD':
            None
        if loss.upper() == 'MSE':
            None

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

        return "Sequential model, {} modules, {} total number of parameters: \n".format(len(self.modules), self.number_params)+'\n'.join(table)