import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
# from torch.autograd import Variable
import numpy as np
import os

import tqdm as tqdm
from statsmodels.tsa.stattools import acf

import math
import time
import random

torch.set_default_tensor_type(torch.DoubleTensor)
from circuitModels import rcModel, rcrModel
from TrivialModels import circuitTrivial
from highdimModels import Highdim
class MH():
    def __init__(self, logden=None, x0=None, std=None):
        self.logden = logden
        self.name = None
        self.x0 = x0
        self.std = std
        self.max_sample = 10000
        self.burn_in = 0.5
        self.thinning = 0.5

    @property
    def jump_dist(self):
        return D.Normal(torch.zeros(self.x0.size(0), self.x0.size(1)), self.std.repeat(self.x0.size(0), 1))
    @property
    def judge_dist(self):
        return D.Uniform(torch.tensor([0.0]).repeat(self.x0.size(0), 1), torch.tensor([1.0]).repeat(self.x0.size(0), 1))

    def sample(self):
        memory, x_curr, total, accept = [], self.x0.clone(), 0, torch.zeros(self.x0.size(0), 1)
        logp = self.logden(x_curr)
        for _ in tqdm.trange(self.max_sample):
            for noise, judge in zip(self.jump_dist.sample([10]), torch.log(self.judge_dist.sample([10]))):
                x_prop, total = noise + x_curr, total + 1
                logq = self.logden(x_prop)
                judge_res = (torch.sum(torch.abs(x_prop) > 4, dim=1, keepdim=True) > 0) + (judge > (logq - logp)) == False
                x_curr[judge_res.reshape(-1), :] = x_prop[judge_res.reshape(-1), :]
                logp[judge_res.reshape(-1), :] = logq[judge_res.reshape(-1), :]
                accept += judge_res
            memory.append(x_curr.clone())
        print("Total Sampled: ", total, "; Accepted: ", accept, "; Accept Rate: ", accept / total)
        return torch.stack(memory, dim=0)

    def post_process(self, m):
        print("Original Size: ", len(m))
        m = m[int(len(m) * self.burn_in):, :]
        m = m[0:len(m):int(1 / self.thinning), :]
        print("Burn_in: ", self.burn_in, "; Thinning: ", self.thinning)
        print("After: ", len(m))
        acfs = []
        for j in range(len(m[0])):
            acfs.append(acf(m[:, j], unbiased=True, nlags=20))
        return m, np.array(acfs)
    def gelman_rubin(self, x):
        M, N = x.size(0), x.size(1)
        mu_m = torch.mean(x, dim=1)
        mu = torch.mean(mu_m, dim=0)
        B = N / (M - 1) * torch.sum((mu_m - mu) ** 2, dim=0)
        W = torch.mean(torch.var(x, dim=1), dim=0)
        R = torch.sqrt(((N - 1) / N * W + (M + 1) / M / N * B) / W)
        return R, torch.cat([x[i] for i in range(x.size(0))], dim=0)

def MH_RC():
    seed = random.randint(1, 10 ** 9)
    print(seed)
    torch.manual_seed(seed)

    cycleTime = 1.07
    totalCycles = 10
    dataSize = 50  # number of observations
    dataname = 'data_rc.txt'
    forcing = np.loadtxt('inlet.flow')
    rc = rcModel(cycleTime, totalCycles, forcing)
    rc.data = np.loadtxt(dataname)
    # rc.surrogate.surrogate_load()

    mh = MH()
    mh.max_sample = 1000
    mh.burn_in = 0.1
    mh.thinning = 1
    mh.logden = lambda x: rc.den_t(x, surrogate=False)

    mh.x0 = torch.Tensor([[0.0, 0.0],
                          [-4.0, -4.0]])
    mh.std = torch.Tensor([0.01, 0.1])
    raw = mh.sample()
    for i in range(mh.x0.size(0)):
        directory = "result/RC_" + str(i)
        if not os.path.isdir(directory): os.makedirs(directory)
        np.savetxt(directory + '/raw_samples.txt', raw[:, i, :].detach().numpy())
        m = np.loadtxt(directory + '/raw_samples.txt')
        m, acfs = mh.post_process(m)
        np.savetxt(directory + '/samples.txt', m)
        np.savetxt(directory + '/acfs.txt', acfs)

def MH_RCR():
    seed = random.randint(1, 10 ** 9)
    print(seed)
    torch.manual_seed(seed)

    cycleTime = 1.07
    totalCycles = 10
    dataSize = 50  # number of observations
    dataname = 'data_rcr.txt'
    forcing = np.loadtxt('inlet.flow')
    rc = rcrModel(cycleTime, totalCycles, forcing)
    rc.data = np.loadtxt(dataname)
    rc.surrogate.surrogate_load()

    mh = MH()
    mh.max_sample = 2000000
    mh.burn_in = 0.1
    mh.thinning = 0.0005
    mh.logden = lambda x: rc.den_t(x, surrogate=True)
    # mh.x0 = torch.Tensor([[1.0, 0.5, -4.0]])
    mh.x0 = torch.Tensor([[-4.0, -4.0, -4.0]])
    mh.std = torch.Tensor([0.025, 0.025, 0.025])

    directory = 'result/RCR_1'
    if not os.path.isdir(directory): os.makedirs(directory)
    mh.x0 = torch.Tensor([[0.0, 0.0, 0.0]])
    mh.std = torch.Tensor([0.025, 0.025, 0.025])
    m = mh.sample()
    np.savetxt(directory + '/rcr_mh.txt', m.detach().numpy())

    m = np.loadtxt(directory + '/rcr_mh.txt')
    m, acfs = mh.post_process(m)
    np.savetxt(directory + '/samples25000', m)
    np.savetxt(directory + '/RCR_acfs', acfs)

def MH_Trivial(directory):
    seed = random.randint(1, 10 ** 9)
    print(seed)
    torch.manual_seed(seed)

    rt = circuitTrivial()
    rt.stdRatio = 0.05
    dataName = 'data_trivial.txt'
    rt.data = np.loadtxt(dataName)

    mh = MH()
    mh.max_sample = 2000000
    mh.burn_in = 0.1
    mh.thinning = 0.001
    mh.logden = lambda x: rt.den_t(x, surrogate=False)
    mh.x0 = torch.Tensor([[5.5, 5.5]])
    mh.std = torch.Tensor([0.01, 0.01])
    m = mh.sample()
    np.savetxt(directory + '/trivial_mh.txt', m.detach().numpy())

    m = np.loadtxt(directory + '/trivial_mh.txt')
    m, acfs = mh.post_process(m)
    np.savetxt(directory + '/samples30000', m)
    np.savetxt(directory + '/Trivial_acfs', acfs)

def MH_Hidim():
    seed = random.randint(1, 10 ** 9)
    print(seed)
    torch.manual_seed(seed)

    hd = Highdim()
    hd.data = np.loadtxt('data_highdim.txt')
    hd.name = "hidim"

    mh = MH()
    mh.max_sample = 30
    mh.burn_in = 0.1
    mh.thinning = 0.1
    mh.logden = lambda x: hd.den_t(x, surrogate=False)


    mh.x0 = torch.Tensor([[3.0, 3.0, 3.0, 3.0, 3.0],
                          [-3.0, -3.0, -3.0, -3.0, -3.0]])
    mh.std = torch.Tensor([0.03, 0.03, 0.03, 0.03, 0.03])
    raw = mh.sample()
    for i in range(mh.x0.size(0)):
        directory = "result/" + hd.name + "_" + str(i)
        if not os.path.isdir(directory): os.makedirs(directory)
        np.savetxt(directory + '/raw_samples.txt', raw[:, i, :].detach().numpy())
        m = np.loadtxt(directory + '/raw_samples.txt')
        m, acfs = mh.post_process(m)
        np.savetxt(directory + '/samples.txt', m)
        np.savetxt(directory + '/acfs.txt', acfs)



if __name__ == "__main__":
    # MH_Hidim()
    MH_RC()
    # MH_RCR()

    # mh = MH()
    # hd = Highdim()
    # hd.data = np.loadtxt('data_highdim.txt')
    # mh.logden = lambda x: hd.den_t(x, surrogate=False)
    #
    # s = torch.stack([torch.Tensor(np.loadtxt("hidim_MH/hidim_0/samples.txt")), torch.Tensor(np.loadtxt("hidim_MH/hidim_1/samples.txt"))],
    #           dim = 0)
    # R, s = mh.gelman_rubin(s)
    # print(R)
    # np.savetxt("samples.txt", s.detach().numpy())
