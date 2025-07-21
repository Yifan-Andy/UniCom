import torch
import torch.nn as nn

class CMMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(CMMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def gaussian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
        L2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)

        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, x, y):
        delta = x.float().mean(0) - y.float().mean(0)
        return delta.dot(delta.T)

    def forward(self, z1, z2, ppr=None, batch_size=None):
        if batch_size is None:
            return self._cmmd_loss(z1, z2, ppr)
        else:
            return self._batched_cmmd_loss(z1, z2, ppr, batch_size)

    def _cmmd_loss(self, source, target, ppr=None):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = source.size(0)
            kernels = self.gaussian_kernel(source, target,
                                           kernel_mul=self.kernel_mul,
                                           kernel_num=self.kernel_num,
                                           fix_sigma=self.fix_sigma)
            if ppr is None:
                XX = torch.mean(kernels[:batch_size, :batch_size])
            else:
                XX = torch.mean(kernels[:batch_size, :batch_size] * ppr)
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            return XX + YY - XY - YX
