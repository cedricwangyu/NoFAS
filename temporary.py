from experiment_setting import experiment
import os
import random
import torch
import numpy as np
import scipy as sp
from maf import MADE, MAF, RealNVP, train
from FNN_surrogate_nested import Surrogate
from TrivialModels import circuitTrivial
from circuitModels import rcModel, rcrModel
from highdimModels import Highdim

print('--- Numpy Version: ', np.__version__)
print('--- Scipy Version: ', sp.__version__)
print('--- Torch Version: ', torch.__version__)

torch.set_default_tensor_type(torch.DoubleTensor)


# Settings


def execute(exp, beta_0=0.5, beta_1=0.1, memory_size=20):
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

    model = model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=exp.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, exp.lr_decay)

    test_mode = 'RCR'
    if test_mode == 'hidim':
        rt = Highdim()
        rt.data = np.loadtxt('source/data/data_highdim.txt')
    elif test_mode == 'trivial':
        rt = circuitTrivial()
        rt.data = np.loadtxt('source/data/data_trivial.txt')
    elif test_mode == 'RC':
        cycleTime = 1.07
        totalCycles = 10
        forcing = np.loadtxt('source/data/inlet.flow')
        rt = rcModel(cycleTime, totalCycles, forcing)  # RC Model Defined
        rt.data = np.loadtxt('source/data/data_rc.txt')
    elif test_mode == 'RCR':
        cycleTime = 1.07
        totalCycles = 10
        forcing = np.loadtxt('source/data/inlet.flow')
        rt = rcrModel(cycleTime, totalCycles, forcing)  # RCR Model Defined
        rt.surrogate = Surrogate("RCR", lambda x: rt.solve_t(rt.transform(x)), rt.numParam, rt.numOutputs,
                                 torch.Tensor([[-7, 7], [-7, 7], [-7, 7]]), memory_size)
        rt.data = np.loadtxt('source/data/data_rcr.txt')
    else:
        raise ValueError('Unrecognized task')
    rt.surrogate.surrogate_load()
    # rt.surrogate.beta_0 = beta_0
    # rt.surrogate.beta_1 = beta_1
    # rt.surrogate.weights = torch.Tensor([np.exp(-rt.surrogate.beta_1 * ii) for ii in range(rt.surrogate.memory_len)])
    loglist = []
    for i in range(exp.n_iter):
        scheduler.step()
        train(model, rt, optimizer, i, exp, loglist, True, True)  # with surrogate
        # train(model, rt, optimizer, i, exp, loglist, True, False) # no surrogate

    # rt.surrogate.surrogate_save() # Used for saving the resulting surrogate model
    np.savetxt(exp.output_dir + '/grid_trace.txt', rt.surrogate.grid_record.detach().numpy())
    np.savetxt(exp.output_dir + '/' + exp.log_file, np.array(loglist), newline="\n")


def post_process(folder):
    cycleTime = 1.07
    totalCycles = 10
    forcing = np.loadtxt('source/data/inlet.flow')
    rt = rcrModel(cycleTime, totalCycles, forcing)
    rt.data = np.loadtxt('source/data/data_rcr.txt')

    params = torch.tensor(np.loadtxt(folder + "/samples25000"))
    params = rt.transform(params)
    np.savetxt(folder + "/RCR_MAF_Parameters.txt", params.detach().numpy())
    print(params)
    model_out = rt.solve_t(params)
    res = torch.normal(0, 1, size=(params.size(0), 3))
    res = model_out + 0.01 * torch.abs(rt.defOut[0]) * res
    print(res)
    np.savetxt(folder + "/RCR_MAF_Samples.txt", res.detach().numpy())


if __name__ == '__main__':
    exp = experiment()
    exp.flow_type = 'maf'  # str: Type of flow                                 default 'realnvp'
    exp.n_blocks = 15  # int: Number of layers                             default 5
    exp.hidden_size = 100  # int: Hidden layer size for MADE in each layer     default 100
    exp.n_hidden = 1  # int: Number of hidden layers in each MADE         default 1
    exp.activation_fn = 'relu'  # str: Actication function used                     default 'relu'
    exp.input_order = 'sequential'  # str: Input order for create_mask                  default 'sequential'
    exp.batch_norm_order = True  # boo: Order to decide if batch_norm is used        default True

    exp.input_size = 3  # int: Dimensionality of input                      default 2
    exp.batch_size = 500  # int: Number of samples generated                  default 100
    exp.true_data_num = 2  # double: number of true model evaluated        default 2
    exp.n_iter = 25001  # int: Number of iterations                         default 25001
    exp.lr = 0.003  # float: Learning rate                              default 0.003
    exp.lr_decay = 0.9999  # float: Learning rate decay                        default 0.9999
    exp.log_interval = 10  # int: How often to show loss stat                  default 10
    exp.calibrate_interval = 300  # int: How often to update surrogate model          default 1000
    exp.budget = 216  # int: Total number of true model evaluation

    exp.output_dir = './results/'
    exp.results_file = 'results.txt'
    exp.log_file = 'log.txt'
    exp.samples_file = 'samples.txt'
    exp.seed = random.randint(0, 1e9)  # int: Random seed used
    print("Seed: ", exp.seed)
    exp.n_sample = 5000  # int: Total number of iterations
    exp.no_cuda = True


    # Single Run
    # exp.output_dir = "./results/" + str(exp.seed) + "/"
    # execute(exp, 0.5, 0.1, 20)
    # post_process(exp.output_dir)
    # filelist = [f for f in os.listdir(exp.output_dir) if
    #             f not in ("grid_trace.txt",
    #                       "log.txt",
    #                       "samples25000",
    #                       "RCR_MAF_Parameters.txt",
    #                       "RCR_MAF_Samples.txt")]
    # for f in filelist:
    #     os.remove(os.path.join(exp.output_dir, f))

    # num_trial = 40
    # seeds = [random.randint(0, 1e9) for _ in range(num_trial)]
    # for trial in range(1, num_trial+1):
    #     print(seeds)
    #     try:
    #         exp.seed = seeds[trial-1]
    #         print(exp.seed)
    #         exp.output_dir = "./result/R_" + str(exp.seed) + "/"
    #         execute(exp, 0.5, 0.1, 20)
    #         post_process(exp.output_dir)
    #     except:
    #         # os.remove(exp.output_dir)
    #         continue
    #
    #     filelist = [f for f in os.listdir(exp.output_dir) if
    #                 f not in ("grid_trace.txt",
    #                           "log.txt",
    #                           "samples25000",
    #                           "RCR_MAF_Parameters.txt",
    #                           "RCR_MAF_Samples.txt")]
    #     for f in filelist:
    #         os.remove(os.path.join(exp.output_dir, f))
    #     print(seeds)

    # for trial in range(1, 2):
    #     for i, beta_0 in enumerate([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
    #         for j, (memory_size, beta_1) in enumerate(zip([20, 20, 9, 2], [0.01, 0.1, 1.0, 10.0])):
    #             try:
    #                 exp.seed = 44803455
    #                 exp.output_dir = "./result/A" + str(i + 1) + "B" + str(j + 1) + "T" + str(trial) + "/"
    #                 execute(exp, beta_0, beta_1, memory_size)
    #                 post_process(exp.output_dir)
    #             except:
    #                 continue
    #
    #             # exp.seed = random.randint(0, 1e9)
    #             # exp.output_dir = "./result/A" + str(i + 1) + "B" + str(j + 1) + "T" + str(trial) + "/"
    #             # execute(exp, beta_0, beta_1, memory_size)
    #
    #
    #             filelist = [f for f in os.listdir(exp.output_dir) if
    #                         f not in ("grid_trace.txt", "log.txt", "samples25000")]
    #             for f in filelist:
    #                 os.remove(os.path.join(exp.output_dir, f))


    # for batch in [1000, 200, 900, 300, 800, 400, 700, 500, 600]:
    #     exp.output_dir = "./results/batch_" + str(batch) + "/"
    #     exp.seed = 735792989
    #     exp.batch_size = batch
    #     try:
    #         execute(exp, 0.5, 0.1, 20)
    #         post_process(exp.output_dir)
    #         filelist = [f for f in os.listdir(exp.output_dir) if
    #                     f not in ("grid_trace.txt",
    #                               "log.txt",
    #                               "samples25000",
    #                               "RCR_MAF_Parameters.txt",
    #                               "RCR_MAF_Samples.txt")]
    #         for f in filelist:
    #             os.remove(os.path.join(exp.output_dir, f))
    #     except:
    #         continue



