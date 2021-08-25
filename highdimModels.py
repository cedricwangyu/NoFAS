import sys, os
import numpy as np
import torch
from FNN_surrogate_nested import Surrogate
torch.set_default_tensor_type(torch.DoubleTensor)


class Highdim:
    def __init__(self):
        # Init parameters
        self.input_num = 5
        self.output_num = 4
        self.x0 = torch.Tensor([0.0838, 0.2290, 0.9133, 0.1524, 0.8258])
        self.defParam = torch.Tensor([[15.6426, 0.2231, 1.2840, 0.0821, 5.7546]])
        self.RM = torch.Tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]) / np.sqrt(2.0)
        self.defOut = self.solve_t(self.defParam)
        self.stdRatio = 0.01
        self.data = None
        self.surrogate = Surrogate("highdim", lambda x: self.solve_t(self.transform(x)), self.input_num, self.output_num,
                                   torch.Tensor([[-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3]]), 20)


    def genDataFile(self, dataSize=50, dataFileName="data_highdim.txt", store=True):
        def_out = self.defOut[0]
        self.data = def_out + self.stdRatio * torch.abs(def_out) * torch.normal(0, 1, size=(dataSize, self.output_num))
        self.data = self.data.t().detach().numpy()
        if store: np.savetxt(dataFileName, self.data)
        return self.data

    def solve_t(self, params):
        return torch.matmul((2 * torch.abs(2 * self.x0 - 1) + params) / (1 + params), self.RM)

    def evalNegLL_t(self, modelOut):
        data_size = len(self.data[0])
        stds = self.defOut * self.stdRatio
        Data = torch.tensor(self.data)
        ll1 = -0.5 * np.prod(self.data.shape) * np.log(2.0 * np.pi)  # a number
        ll2 = (-0.5 * self.data.shape[1] * torch.log(torch.prod(stds))).item()  # a number
        # ll3 = 0.0
        # for i in range(self.output_num):
        #     ll3 = ll3 - 0.5 * torch.sum(((modelOut[:, i].repeat(data_size, 1).t() - Data[i, :]) / stds[0, i]) ** 2,
        #                                 dim=1)
        ll3 = - 0.5 * torch.sum(torch.sum((modelOut.unsqueeze(0) - Data.t().unsqueeze(1)) ** 2, dim=0) / stds[0] ** 2, dim=1, keepdim=True)
        negLL = -(ll1 + ll2 + ll3)
        return negLL

    def transform(self, x):
        return torch.exp(x)

    def den_t(self, x, surrogate=True):
        batch_size = x.size(0)
        adjust = torch.sum(x, dim=1, keepdim=True)
        if surrogate:
            modelOut = self.surrogate.forward(x)
        else:
            modelOut = self.solve_t(self.transform(x))
        return - self.evalNegLL_t(modelOut).reshape(batch_size, 1) + adjust
    def rev_solve_t(self, y):
        x = torch.Tensor([[0]] * y.size(0))
        x = torch.cat([y[:, 3:4], x], dim=1)
        x = torch.cat([y[:, 2:3] - x[:, 0:1], x], dim=1)
        x = torch.cat([y[:, 1:2] - x[:, 0:1], x], dim=1)
        x = torch.cat([y[:, 0:1] - x[:, 0:1], x], dim=1) * np.sqrt(2)

        con = 2 * torch.abs(2 * self.x0 - 1)
        print(con)
        t1 = (1 - x) / torch.Tensor([1, -1, 1, -1, 1])
        t2 = (con - x) / torch.Tensor([1, -1, 1, -1, 1])
        tmin = torch.cat([t1[:, 0:1], t2[:, 1:2], t1[:, 2:3], t2[:, 3:4], t1[:, 4:5]], dim=1)
        tmax = torch.cat([t2[:, 0:1], t1[:, 1:2], t2[:, 2:3], t1[:, 3:4], t2[:, 4:5]], dim=1)
        tmin = torch.max(tmin, dim=1, keepdim=True)[0]
        tmax = torch.min(tmax, dim=1, keepdim=True)[0]
        return x, torch.cat([tmin, tmax], dim=1)

def Test_Highdim():
    # Define Class and parameters
    hd = Highdim()
    dataSize = 50
    dataName = 'data_highdim.txt'

    # Define Data
    if False:
        hd.genDataFile(dataSize, dataName)

    # Load Data
    hd.data = np.loadtxt(dataName)
    # print(hd.data.shape)
    # Test Model Solution and Neg_Log_Likelihood
    if False:
        params = torch.Tensor([[15.6426, 0.2231, 1.2840, 0.0821, 5.7546],
                               [15.6426, 15.6426, 15.6426, 15.6426, 15.6426],
                               [0.2231, 0.2231, 0.2231, 0.2231, 0.2231],
                               [1.2840, 1.2840, 1.2840, 1.2840, 1.2840],
                               [0.0821, 0.0821, 0.0821, 0.0821, 0.0821],
                               [5.7546, 5.7546, 5.7546, 5.7546, 5.7546]])
        print('parameters: \n', params)
        out = hd.solve_t(params)
        print('Model results: \n', out)

        ll = hd.evalNegLL_t(out)
        print('Model log-likelihood: \n', ll)
    # Test Model den_t
    if False:
        params = torch.Tensor([[2.75, -1.5, 0.25, -2.5, 1.75],
                               [2.75, 2.75, 2.75, 2.75, 2.75],
                               [-1.5, -1.5, -1.5, -1.5, -1.5],
                               [0.25, 0.25, 0.25, 0.25, 0.25],
                               [-2.5, -2.5, -2.5, -2.5, -2.5],
                               [1.5506, 14.3978, -0.1642, -1.2035,  0.6458]])
        # print('parameters: \n', params)
        # print(hd.solve_t(hd.transform(params)))
        den = hd.den_t(params, surrogate=False)
        print('Den_t: \n', den)
    # Plot Solutions and Neg_Likelihood
    if False:
        gridnum = 30  # Grid size for one dim
        # limits = [[4.94, 5.1], [9.96, 10.06]]
        limits = [[-1, 3], [-2, 2]]
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
        np.savetxt('Trivial_LL.txt', ll.detach().numpy())
        exit(-1)

    # Train surrogate with pregrid
    if False:
        hd.surrogate.gen_grid(input_limits=None, gridnum=10, store=True)
        hd.surrogate.pre_train(60000, 0.03, 0.9999, 500, store=True, reg=True)
        hd.surrogate.surrogate_save()
        # hd.surrogate.surrogate_load()
        # print(hd.surrogate.pre_grid)
        # print(hd.surrogate.pre_out)
        # loss = torch.norm(hd.surrogate.pre_out - hd.surrogate.forward(hd.surrogate.pre_grid), dim=1, keepdim=True) / torch.norm(hd.surrogate.pre_out, dim=1, keepdim=True)
        # loss = torch.norm(hd.surrogate.pre_out - hd.surrogate.forward(hd.surrogate.pre_grid), dim=1, keepdim=True)
        # print(loss)
        # np.savetxt('fix_oloss_pre_4p5.txt', loss.detach().numpy())


    # Check surrogate on its own grid
    if False:
        hd.surrogate.surrogate_load()
        print(hd.surrogate.grid_record)
        true_out = hd.solve_t(hd.transform(hd.surrogate.grid_record))
        # print(true_out)
        surr_out = hd.surrogate.forward(hd.surrogate.grid_record)
        # loss = torch.norm(true_out - surr_out, dim=1, keepdim=True) / torch.norm(true_out, dim=1, keepdim=True)
        loss = torch.norm(true_out - surr_out, dim=1, keepdim=True)
        # print(loss)
        np.savetxt('update_flat_lloss_post_3p5.txt', loss.detach().numpy())
    # Check surrogate on a finer grid
    if False:
        hd.surrogate.surrogate_load()
        gridnum = 20  # Grid size for one dim
        limits = [[-3, 3], [-3, 3],[-3, 3],[-3, 3],[-3, 3]]
        meshpoints = [torch.linspace(l[0], l[1], steps=gridnum) for l in limits]
        grid = torch.meshgrid(meshpoints)
        gridlist = torch.cat([item.reshape(gridnum ** len(limits), 1) for item in grid], 1)
        # loss = torch.norm(hd.solve_t(hd.transform(gridlist)) - hd.surrogate.forward(gridlist), dim=1, keepdim=True) / torch.norm(hd.solve_t(hd.transform(gridlist)), dim=1, keepdim=True)
        loss = torch.norm(hd.solve_t(hd.transform(gridlist)) - hd.surrogate.forward(gridlist), dim=1, keepdim=True)
        # print(loss)
        np.savetxt('flat_update_lloss_post_20p5.txt', loss.detach().numpy())

    # Generate Solution
    if False:
        curve = []
        x0 = np.array([0.0838, 0.2290, 0.9133, 0.1524, 0.8258])
        y0 = np.array([1.039945681564179, 1.068677949472652, 1.285989492119089, 1.360779964883098, 1.044887928226690])
        t = -0.0153
        while t <= 0.0686:
            y = y0 + np.array([1, -1, 1, -1, 1]) * t;
            curve.append(list(np.log((2 * abs(2 * x0 - 1) - y) / (y - 1))))
            t += 0.00002
        curve = torch.Tensor(curve)

        hd.surrogate.surrogate_load()
        m = torch.min(torch.norm(curve.unsqueeze(0) - hd.surrogate.grid_record.unsqueeze(1), dim=2), dim=1)[0]
        print(list(m.size()))
        np.savetxt('pregrid-d.txt', m.detach().numpy())

        # with torch.no_grad():
        #     gridnum = 20  # Grid size for one dim
        #     limits = [[-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3]]
        #     meshpoints = [torch.linspace(l[0], l[1], steps=gridnum) for l in limits]
        #     grid = torch.meshgrid(meshpoints)
        #     gridlist = torch.cat([item.reshape(gridnum ** len(limits), 1) for item in grid], 1)
        #     m = torch.Tensor([])
        #     for i in range(320):
        #         print(i)
        #         m = torch.cat([m, torch.min(torch.norm(curve.unsqueeze(0) - gridlist[10000 * i: 10000 * (i+1), :].unsqueeze(1), dim=2), dim=1)[0]])
        #     print(list(m.size()))
        #     np.savetxt('test-grid-d.txt', m.detach().numpy())
    # Solve MAF sample parameters
    if True:
        # params = torch.tensor(np.loadtxt('samples25000'))
        params = torch.tensor(np.loadtxt('samples.txt'))
        params = hd.transform(params)
        # np.savetxt('hidim_MAF_Parameters.txt', params.detach().numpy())
        np.savetxt('hidim_MH_Parameters.txt', params.detach().numpy())
        print(params)
        print(hd.defOut[0])
        model_out = hd.solve_t(params)
        # np.savetxt('hidim_MAF_Samples_mean.txt', model_out.detach().numpy())
        # print(model_out)
        res = torch.normal(0, 1, size=(params.size(0), hd.output_num))
        res = model_out + hd.stdRatio * torch.abs(hd.defOut[0]) * res
        print(res)
        # np.savetxt('hidim_MAF_Samples.txt', res.detach().numpy())
        np.savetxt('hidim_MH_Samples.txt', res.detach().numpy())


# TEST MODELS
if __name__ == "__main__":
    Test_Highdim()
    #
    # hd = Highdim()

    # y = torch.Tensor(np.loadtxt("data_highdim.txt")).t()
    # y = hd.defOut
    # x, t = hd.rev_solve_t(y)
    # print(x + torch.Tensor([1,-1,1,-1,1]) * t[0,0], x + torch.Tensor([1,-1,1,-1,1]) * t[0,1])
    # np.savetxt("special_x.txt", x)
    # np.savetxt("t_range.txt", t)

