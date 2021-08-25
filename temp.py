from experiment_setting import experiment
import os
import torch
import numpy as np
import scipy as sp
from maf import MADE, MAF, RealNVP, train
import time
import random
from TrivialModels import circuitTrivial
from circuitModels import rcModel, rcrModel

print('--- Numpy Version: ', np.__version__)
print('--- Scipy Version: ', sp.__version__)
print('--- Torch Version: ', torch.__version__)

torch.set_default_tensor_type(torch.DoubleTensor)
start_time = time.time()

# Settings
exp = experiment()
exp.flow_type = 'realnvp'                   # str: Type of flow                                 default 'maf'
exp.n_blocks = 5                       # int: Number of layers                             default 64
exp.hidden_size = 100                    # int: Hidden layer size for MADE in each layer     default 32
exp.n_hidden = 1                        # int: Number of hidden layers in each MADE         default 1
exp.activation_fn = 'relu'              # str: Actication function used                     default 'relu'
exp.input_order = 'sequential'          # str: Input order for create_mask                  default 'sequential'
exp.batch_norm_order = True             # boo: Order to decide if batch_norm is used        default True

exp.input_size = 2                      # int: Dimensionality of input                      default 2
exp.batch_size = 200                    # int: Number of samples generated                  default 100
exp.true_data_num = 2               # double: proportion of true model evaluated        default 0.3
exp.n_iter = 30001                     # int: Number of iterations                         default 3000
exp.lr = 0.002                          # float: Learning rate                              default 0.003, Trivial 0.03
exp.lr_decay = 0.9999                   # float: Learning rate decay                        default 0.9999, Trivial 0.9995
exp.log_interval = 10                   # int: How often to show loss stat                  default 10
exp.calibrate_interval = 1000
exp.budget = 64



for iteration in range(1):
    exp.output_dir = './results/trivial_' +str(iteration)
    exp.results_file = 'results.txt'
    exp.log_file = 'log.txt'
    exp.samples_file = 'samples.txt'

    exp.seed = random.randint(1, 10 ** 9)
    exp.n_sample = 5000
    exp.no_cuda = True
    # redo = True

    # setup file ops
    if not os.path.isdir(exp.output_dir):
        os.makedirs(exp.output_dir)

    # setup device
    print("Cuda Availability: ", torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')
    torch.manual_seed(exp.seed)
    if device.type == 'cuda': torch.cuda.manual_seed(exp.seed)

    # model
    if exp.flow_type == 'made':
        model = MADE(exp.input_size, exp.hidden_size, exp.n_hidden, None, exp.activation_fn, exp.input_order)
    elif exp.flow_type == 'maf':
        model = MAF(exp.n_blocks, exp.input_size, exp.hidden_size, exp.n_hidden, None,
                    exp.activation_fn, exp.input_order, batch_norm=exp.batch_norm_order)
    elif exp.flow_type == 'realnvp':  # Under construction
        model = RealNVP(exp.n_blocks, exp.input_size, exp.hidden_size, exp.n_hidden, None,
                        batch_norm=exp.batch_norm_order)
    else:
        raise ValueError('Unrecognized model.')
    # for name, param in model.named_parameters():
    #     print(name, param)
    model = model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=exp.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, exp.lr_decay)

    test_mode = 'trivial'
    if test_mode == 'trivial':
        rt = circuitTrivial()
        rt.data = np.loadtxt('data_trivial.txt')
        rt.surrogate.surrogate_load()
        print(rt.surrogate.pre_grid)
        print(rt.surrogate.pre_out)

        # Test for pre_out
        if False:
            np.savetxt('grid_trace.txt', rt.surrogate.pre_grid.detach().numpy())
            exit(-1)
    elif test_mode == 'RC':
        cycleTime = 1.07
        totalCycles = 10
        forcing = np.loadtxt('inlet.flow')
        rt = rcModel(cycleTime, totalCycles, forcing)  # RC Model Defined
        rt.data = np.loadtxt('data_rc.txt')
        rt.surrogate.surrogate_load()
        if False:
            # print(rt.surrogate.pre_grid)
            # print(rt.den_t(rt.surrogate.pre_grid))
            np.savetxt("den_t.txt", rt.den_t(rt.surrogate.pre_grid).detach().numpy())
            exit(-1)
    elif test_mode == 'RCR':
        cycleTime = 1.07
        totalCycles = 10
        forcing = np.loadtxt('inlet.flow')
        rt = rcrModel(cycleTime, totalCycles, forcing)  # RCR Model Defined
        rt.data = np.loadtxt('data_rcr.txt')
        rt.surrogate.surrogate_load()
        if False:
            # print(rt.surrogate.pre_grid)
            # print(rt.den_t(rt.surrogate.pre_grid))
            np.savetxt("den_t.txt", rt.den_t(rt.surrogate.pre_grid).detach().numpy())
            exit(-1)

    loglist = []
    for i in range(exp.n_iter):
        scheduler.step()
        train(model, rt, optimizer, i, exp, loglist, True, True)
    # histo = torch.cat([torch.flatten(param) for param in model.parameters()])
    # np.savetxt(exp.output_dir + '/params', histo.detach().numpy(), newline="\n")
    # rt.surrogate.surrogate_save()
    np.savetxt(exp.output_dir + '/' + 'grid_trace.txt', rt.surrogate.grid_record.detach().numpy())
    np.savetxt(exp.output_dir + '/' + exp.log_file, np.array(loglist), newline="\n")
    print("%s seconds" % (time.time() - start_time))























