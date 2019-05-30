import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            '''Your codes here'''
            mu = self.running_mean * self.momentum + (1 - self.momentum) * torch.mean(x, 0).data
            var = self.running_var * self.momentum + (1 - self.momentum) * torch.var(x, 0).data
            xnorm = (x - mu) / torch.sqrt(var + self.eps)
            self.running_mean = mu
            self.running_var = var
            return self.weight * xnorm + self.bias

        else:
            '''Your codes here'''
            mu = self.running_mean
            var = self.running_var
            xnorm = (x - mu) / torch.sqrt(var + self.eps)
            return self.weight * xnorm + self.bias


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            '''Your codes here'''
            xnew = torch.zeros(x.size())
            mu = torch.zeros(self.num_features)
            var = torch.ones(self.num_features)
            for i in range(self.num_features):
                mu[i] = self.running_mean[i] * self.momentum + (1 - self.momentum) * torch.mean(x[:, i, :, :]).data
                var[i] = self.running_var[i] * self.momentum + (1 - self.momentum) * torch.var(x[:, i, :, :]).data
                temp = (x[:, i, :, :] - self.running_mean[i]) / torch.sqrt(self.running_var[i] + self.eps)
                xnew[:, i, :, :] = self.weight[i] * temp + self.bias[i]
            self.running_mean = mu
            self.running_var = var
            return xnew

        else:
            '''Your codes here'''
            xnew = torch.zeros(x.size())
            mu = torch.zeros(self.num_features)
            var = torch.ones(self.num_features)
            for i in range(self.num_features):
                mu[i] = self.running_mean[i]
                var[i] = self.running_var[i]
                temp = (x[:, i, :, :] - mu[i]) / torch.sqrt(var[i] + self.eps)
                xnew[:, i, :, :] = self.weight[i] * temp + self.bias[i]
            return xnew


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
