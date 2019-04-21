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
            self.running_mean = torch.mean(x, 0)
            self.running_var = torch.var(x, 0)
            xnorm = (x - self.running_mean) / torch.sqrt(self.running_var + eps)
            return self.weight * xnorm + self.bias

        else:
            '''Your codes here'''
            mu = self.running_mean * self.momentum + (1 - self.momentum) * x
            var = self.running_var * self.momentum + (1 - self.momentum) * x
            xnorm = (x - mu) / torch.sqrt(var + eps)
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
            self.running_mean = torch.mean(x, (0,2,3))
            self.running_var = torch.var(x, (0,2,3))
            xnorm = (x - self.running_mean) / torch.sqrt(self.running_var + eps)
            return self.weight * xnorm + self.bias

        else:
            '''Your codes here'''
            pass


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
