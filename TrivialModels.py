import sys, os
import numpy as np
import torch
from FNN_surrogate_nested import Surrogate


torch.set_default_tensor_type(torch.DoubleTensor)


class circuitTrivial:
    def __init__(self):
        # Init parameters
        # self.defParam = torch.Tensor([[5.0, 10.0]])
        # self.defParam = torch.Tensor([[0.5, 0.5]])
        self.defParam = torch.Tensor([[3.0, 5.0]])
        self.RM = torch.Tensor([[1.0, 1.0],
                                [1.0, -1.0]])
        self.stdRatio = 0.05
        self.surrogate = Surrogate("Trivial", self.solve_t, 2, 2, [[0, 6], [0, 6]], 20)
        self.data = None

    def genDataFile(self, dataSize=50, dataFileName="data_trivial.txt", store=True):
        def_out = self.solve_t(self.defParam)[0]
        print(def_out)
        self.data = def_out + self.stdRatio * torch.abs(def_out) * torch.normal(0, 1, size=(dataSize, 2))
        # self.data = def_out + self.stdRatio * torch.normal(0, 1, size=(dataSize, 2))
        self.data = self.data.t().detach().numpy()
        if store: np.savetxt(dataFileName, self.data)
        return self.data

    def solve_t(self, params):
        z1, z2 = torch.chunk(params, chunks=2, dim=1)
        x = torch.cat((z1 ** 3 / 10, torch.exp(z2 / 3)), 1)
        # x = torch.cat((z1 - 1, (z2 - 1) ** 2), 1) / 2
        # x = torch.cat((z1, z2 ** 2), 1) / 2
        return torch.matmul(x, self.RM)

    def evalNegLL_t(self, params, surrogate=True):
        data_size = len(self.data[0])
        stds = torch.abs(self.solve_t(self.defParam)) * self.stdRatio
        # stds = self.stdRatio * torch.ones(1, 2)
        if not surrogate:
            modelOut = self.solve_t(params)
        else:
            modelOut = self.surrogate.forward(params)
        Data = torch.tensor(self.data)
        # Eval LL
        ll1 = -0.5 * np.prod(self.data.shape) * np.log(2.0 * np.pi)  # a number
        ll2 = (-0.5 * self.data.shape[1] * torch.log(torch.prod(stds))).item()  # a number
        ll3 = 0.0

        for i in range(2):
            ll3 = ll3 - 0.5 * torch.sum(((modelOut[:, i].repeat(data_size, 1).t() - Data[i, :]) / stds[0, i]) ** 2,
                                        dim=1)
        negLL = -(ll1 + ll2 + ll3)
        return negLL

    def den_t(self, params, surrogate=True):
        return - self.evalNegLL_t(params, surrogate)


def Test_Trivial():
    # Define Class and parameters
    rt = circuitTrivial()
    dataSize = 50
    rt.stdRatio = 0.05
    dataName = 'data_trivial.txt'

    # Define Data
    if False:
        rt.genDataFile(dataSize, dataName)

    # Load Data
    rt.data = np.loadtxt(dataName)

    # Test Model Solution and Neg_Log_Likelihood
    if False:
        params = torch.Tensor([[0, 5],
                               [0, 10],
                               [0, 15],
                               [5, 5],
                               [5, 10],
                               [5, 15],
                               [10, 5],
                               [10, 10],
                               [10, 15]])

        print('parameters: \n', params)
        out = rt.solve_t(params)
        print('Model results: \n', out)

        ll = rt.evalNegLL_t(params)
        print('Model log-likelihood: \n', ll)

    # Plot Solutions and Neg_Likelihood
    if False:
        gridnum = 30  # Grid size for one dim
        limits = [[2.75, 3.2], [4.87, 5.05]]
        # limits = [[0, 6], [0, 6]]
        meshpoints = []
        meshpoints.append(torch.linspace(limits[0][0], limits[0][1], steps=gridnum))
        meshpoints.append(torch.linspace(limits[1][0], limits[1][1], steps=gridnum))

        grid = torch.meshgrid(meshpoints)
        gridlist = torch.cat([item.reshape(gridnum ** len(limits), 1) for item in grid], 1)
        print(gridlist)
        modelout = rt.solve_t(gridlist)
        print(modelout)
        # Save the True Solution of RC
        # np.savetxt('Trivial_Solution.txt', modelout.detach().numpy())

        # Evaluate Neg_Log_Likelihood
        ll = - rt.evalNegLL_t(gridlist, surrogate=False)
        print(ll)
        # np.savetxt('Trivial_LL.txt', ll.detach().numpy())
        np.savetxt('Trivial_LL_small_a.txt', ll.detach().numpy())
        exit(-1)
    # Train surrogate with pregrid
    if False:
        S = Surrogate("Trivial", rt.solve_t, 2, 2, [[0, 6], [0, 6]], 20)
        S.gen_grid(input_limits=None, gridnum=30, store=True)
        S.pre_train(60000, 0.03, 0.9999, 500, store=True, reg=True)
        S.surrogate_save()
        # S.surrogate_load()

    # Solve MAF sample parameters
    if False:
        params = torch.tensor(np.loadtxt("samples30000"))
        np.savetxt('Trivial_MAF_Parameters.txt', params.detach().numpy())
        rt.stdRatio = 0.05
        def_out = rt.solve_t(rt.defParam)[0]
        model_out = rt.solve_t(params)
        res = torch.normal(0, 1, size=(list(params.size())[0], 2))
        res = model_out + rt.stdRatio * torch.abs(def_out) * res
        print(res)
        np.savetxt('Trivial_MAF_Samples.txt', res.detach().numpy())

    if False:
        rt.surrogate.surrogate_load()
        gridnum = 500  # Grid size for one dim
        limits = [[-4, 10], [0, 15]]
        grid = torch.meshgrid([torch.linspace(limits[0][0], limits[0][1], steps=gridnum),
                               torch.linspace(limits[1][0], limits[1][1], steps=gridnum)])

        grid = torch.cat([item.reshape(gridnum ** len(limits), 1) for item in grid], 1)
        np.savetxt('nest_pos_true.txt', -rt.evalNegLL_t(grid, surrogate=False).detach().numpy())
        np.savetxt('nest_pos_error.txt', torch.abs(rt.evalNegLL_t(grid, surrogate=False)
                                                   - rt.evalNegLL_t(grid, surrogate=True)).detach().numpy())
        print("Error at true parameter: ", torch.abs(
            rt.evalNegLL_t(torch.Tensor([[5, 10]]), surrogate=False) - rt.evalNegLL_t(torch.Tensor([[5, 10]]),
                                                                                      surrogate=True)))
        grid_trace = torch.Tensor(np.loadtxt('grid_trace.txt'))
        np.savetxt('grid_trace_error.txt', torch.abs(rt.evalNegLL_t(grid_trace, surrogate=False)
                                                     - rt.evalNegLL_t(grid_trace, surrogate=True)).detach().numpy())
        np.savetxt('grid_trace_true.txt', -rt.evalNegLL_t(grid_trace, surrogate=False).detach().numpy())


# TEST MODELS
if __name__ == "__main__":
    Test_Trivial()
