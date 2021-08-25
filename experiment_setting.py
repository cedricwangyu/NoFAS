import os

class experiment():
    def __init__(self):
        self.flow_type          = 'made'         # str: Type of flow
        self.n_blocks           = 5             # int: Number of layers
        self.hidden_size        = 100           # int: Hidden layer size for MADE in each layer
        self.n_hidden           = 1             # int: Number of hidden layers in each MADE
        self.activation_fn      = 'relu'        # str: Actication function used
        self.input_order        = 'sequential'  # str: Input order for create_mask
        self.batch_norm_order   = True          # boo: Order to decide if batch_norm is used

        self.input_size         = 2
        self.batch_size         = 100           # int: Number of samples generated
        self.true_data_num      = 2             # double: proportion of true model evaluated
        self.n_iter             = 1e4           # int: Number of iterations
        self.lr                 = 1e-4          # float: Learning rate
        self.lr_decay           = 0.999         # float: Learning rate decay
        self.log_interval       = 300           # int: How often to show loss sta
        self.calibrate_interval = 1000
        self.budget             = 64

        self.output_dir         = './results/{}'.format(os.path.splitext(__file__)[0])
        self.results_file       = 'results.txt'
        self.log_file           = 'log.txt'
        self.samples_file       = 'samples.txt'
        self.seed               = 1
        self.n_sample           = 10000
        self.no_cuda            = True



