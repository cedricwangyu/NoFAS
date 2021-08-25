import torch
from torch.nn import functional as F
import torch.distributions as D
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)
def analytic_1(z):
    # Analytical Density: Four modal at center (rm, um)
    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    rm = 0
    um = 0
    norm = torch.sqrt((z1 - rm) ** 2 + (z2 - um) ** 2)

    exp1 = torch.exp(-0.5 * (((z1 - rm) - 2) / 0.8) ** 2)
    exp2 = torch.exp(-0.5 * (((z1 - rm) + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - torch.log(exp1 + exp2)

    return torch.exp(-u)

def analytic_2(z):
    # Analytical Density: I just randomly play with analytic_1 and see what happens - but still it is analytic
    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    norm = torch.sqrt((z1 + 6) ** 2 + (z2 + 6) ** 2)

    exp1 = torch.exp(-0.5 * (((z1 + 6) - 1) / 0.8) ** 2)
    exp2 = torch.exp(-0.5 * (((z2 + 6) + 3) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.8) ** 2 - torch.log(exp1 + exp2 + 0.6)

    return torch.exp(-u)

def exam(z):
    # Analytical Density: Uniform Blocks that to approximate RC true posterior - No Smoothing Edges
    n = list(z.size())[0]
    z1, z2 = torch.chunk(z, chunks = 2, dim = 1)
    # region1 = torch.logical_or(torch.logical_and(torch.logical_and(z1 > 0, z1 < 1), z2 < 0), torch.logical_and(torch.logical_and(z2 < 0, z2 > -1), z1 > 0))
    region1 = torch.logical_and(torch.logical_and(z1 > 0, z1 < 2), torch.logical_and(z2 < 0, z2 > -1))
    region2 = torch.logical_and(torch.logical_and(z1 > 0, z1 < 1), torch.logical_and(z2 <= -1, z2 > -2))
    region3 = torch.logical_and(torch.logical_and(z1 >= 1, z1 < 2), torch.logical_and(z2 <= -1, z2 > -2))
    region4 = torch.logical_and(torch.logical_and(z1 > 0, z1 < 2), torch.logical_and(z2 >= 0, z2 < 2))
    # region2 = torch.logical_and(z1 >= 0, z2 >= 0)
    # region3 = torch.logical_and(z1 >= 1, z2 <= -1)
    
    # print(region1 * 8 + region2 * 7 + region3 * 5)
    return region1 * 8 + region2 * 7 + region3 * 5
    # return (region1 + region2) * 8 + region3 * 7 + region4 * 5 + (1 - region1 * 1) * (1 - region2 * 1) * (1 - region3 * 1) * (1 - region4 * 1) * torch.exp( - 3 * torch.sum((z - 1) ** 2, dim = 1).reshape(n,1))

def exam_spline(z):
    # Analytical Density: Uniform Blocks that to approximate RC true posterior - Linearly Smoothing Edges
    def squareden(z1, z2, left, right, up, down, level, rate = 0.1):
        region = torch.logical_and(torch.logical_and(z1 > left, z1 < right), torch.logical_and(z2 > down, z2 < up))
        # Sub-function that used to build an uniform block with linearly smoothed edges
        LEFT = (left+right)/2 - (right-left)/2*(1+rate)
        RIGHT = (left+right)/2 + (right-left)/2*(1+rate)
        DOWN = (up+down)/2 - (up-down)/2*(1+rate)
        UP = (up+down)/2 + (up-down)/2*(1+rate)

        region_out = torch.logical_and(torch.logical_and(z1 > LEFT, z1 < RIGHT), torch.logical_and(z2 > DOWN, z2 < UP))
        region_inter = torch.logical_and(region_out, torch.logical_not(region)) \
                       * (1 - torch.max(torch.cat([(z1 - right)/(RIGHT-right), (left - z1)/(RIGHT-right), (z2 - up)/(UP-up), (down - z2)/(UP-up)], dim = 1), dim = 1, keepdim=True)[0])
        return (region + region_inter) * level

    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    region1 = squareden(z1, z2, 0, 2, 0, -1, 8)
    region2 = squareden(z1, z2, 0, 1, -1, -2, 8)
    region3 = squareden(z1, z2, 1, 2, -1, -2, 7)
    region4 = squareden(z1, z2, 0, 2, 2, 0, 5)
    return torch.max(torch.cat([region1, region2, region3, region4], dim = 1), dim = 1, keepdim = True)[0]

def exam_spline_3(z):
    # Analytical Density: Naive Augment Idea with exam_spline. z2 is the augmented dimension
    z1, z2 = torch.split(z, [2,1], dim = 1)
    return exam_spline(z1) + torch.exp(-z2 ** 2)



def uniform_circle(z):
    # Analytical Density: Uniform Distribution on a disk with exponential smoothing
    n = list(z.size())[0]
    region = torch.sum((z - 1) ** 2, dim = 1) < 4
    region = region.reshape(n, 1) * 1
    return region + (1 - region) * torch.exp((-torch.sum((z - 1) ** 2, dim = 1).reshape(n,1) + 4) * 20)

def uniform_circle_t(z):
    # Analytical Density: Uniform Distribution on a disk with linearly smoothing edges
    n = list(z.size())[0]
    dis = torch.sum((z - 1) ** 2, dim = 1).reshape(n, 1)
    region = dis < 4
    region = region.reshape(n, 1) * 1
    region2 = torch.logical_and(dis < 4.5, dis > 4)
    region2 = region2.reshape(n, 1) * 1
    return region + region2 * (4.5 - dis) * 2


class Mixed_Gaussian():
    # Class for building Mixed Gaussian with weight w, mean m, cov matrix s
    def __init__(self, w, m, s, size):
        n = list(m.size())
        self.dim = n[1]
        self.n = n[0]
        self.size = size
        self.m = m
        self.M = m.reshape(self.n, 1, self.dim).repeat(1, self.size, 1)
        self.s = s
        self.S = s.reshape(self.n, 1).repeat(1, self.size)
        self.w = w
        self.W = w.reshape(1, self.n)

    def density(self, x, log = False):
        X = x.repeat(self.n, 1, 1)
        res = torch.mm(self.W, torch.exp(torch.sum(- (X - self.M) ** 2, 2) / self.S ** 2 / 2) / self.S ** self.dim).reshape(self.size)
        if log:
            return torch.log(res)
        else:
            return res
        
    def sample(self, n):
        normal = D.Normal(0,1)
        x = normal.sample([n, self.dim])
        I = 0
        tw = self.w[I]
        cut = int(tw * n) - 1
        for i in range(n):
            if i > cut:
                I += 1
                tw += self.w[I]
                cut = int(tw * n) - 1
                
            x[i] = x[i] * self.s[I] + self.m[I]

        return x

    def corr(self):
        mu = sum(self.m * self.w.repeat(self.dim, 1).t())
        sigma = torch.sqrt(torch.sum((self.s ** 2 + self.m.t() ** 2) * self.w, 1) - mu ** 2)
        cov = torch.mm(m.t() * w,m) - torch.mm(mu.reshape(self.dim,1), mu.reshape(1,self.dim))
        corr = cov / torch.mm(sigma.reshape(self.dim,1), sigma.reshape(1,self.dim))
        for i in range(self.dim):
            corr[i,i] = 1

        return corr






if __name__ == '__main__':
    if False:
        # Testing Mixed Gaussian
        Standard_Normal = D.Normal(0, 1)
        w = torch.Tensor([0.2,0.1,0.3,0.3,0.1])
        m = torch.Tensor([[0.4,1.5,-1.9,-0.2],[1.5,-1.9,0.7,-0.2],[-0.1,-1.1,0.3,-1.2],[0.6,0.1,0.9,0.1],[0.5,-0.8,-1.5,-1.2]])
        s = torch.Tensor([0.4, 0.2, 0.8, 0.5, 0.4])
        batch_size = 5
        Dist = Mixed_Gaussian(w, m, s, batch_size)
        print(Dist.corr())
        x = torch.Tensor(m)
        print(Dist.density(x))
        x = Dist.sample(30000)
        np.savetxt("/Users/yuwang/Documents/GitHub/Modern_NF/sample_mixed_Gaussian.txt", np.array(x), newline = "\n")

    if False:
        # Plotting Exam & Exam_spline function on a grid
        meshpoints = []
        limits = [[-3, 3], [-3, 3]]
        gridnum = 100
        meshpoints.append(torch.linspace(limits[0][0], limits[0][1], steps = gridnum))
        meshpoints.append(torch.linspace(limits[1][0], limits[1][1], steps = gridnum))
        grid = torch.meshgrid(meshpoints)
        gridlist = torch.cat([item.reshape(gridnum ** len(limits) ,1) for item in grid], 1)

        res = exam(gridlist)
        # res = exam_spline(gridlist)
        print(res)
        np.savetxt('testing_exam_m.txt', res.detach().numpy())
        # np.savetxt('testing_exam_spline.txt', res.detach().numpy())
