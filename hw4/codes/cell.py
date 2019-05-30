import torch.nn as nn
import torch
import math


class RNNCell(nn.Module):
    '''An Elman RNN cell with tanh non-linearity.

    .. math::

        h' = \tanh(x w_{ih} + b_{ih}  +  h w_{hh} + b_{hh})

    Inputs: input, h
        - **input** of shape `(batch, input_dim)`: tensor containing input features
        - **h** of shape `(batch, hidden_dim)`: tensor containing the initial
        hidden state for each element in the batch.

    Outputs: h'
        - **h'** of shape `(batch, hidden_dim)`: tensor containing the next hidden state
          for each element in the batch

    '''
    def __init__(self, input_dim, hidden_dim):
        super(RNNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w_ih = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.w_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_dim))
        self.reset_params()

    def reset_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, h):
        # TODO: your codes here
        h_new = torch.tanh(torch.matmul(input, self.w_ih) + self.b_ih + torch.matmul(h[0], self.w_hh) + self.b_hh)
        return h_new, h_new


class GRUCell(nn.Module):
    '''A gated recurrent unit (GRU) cell

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    where :math:`\sigma` is the sigmoid function.

    Inputs: input, h
        - **input** of shape `(batch, input_dim)`: tensor containing input features
        - **h** of shape `(batch, hidden_dim)`: tensor containing the initial
        hidden state for each element in the batch.

    Outputs: h'
        - **h'** of shape `(batch, hidden_dim)`: tensor containing the next hidden state
          for each element in the batch
    '''
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w_ir = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.w_hr = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_ir = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_hr = nn.Parameter(torch.Tensor(hidden_dim))

        self.w_iz = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.w_hz = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_iz = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_hz = nn.Parameter(torch.Tensor(hidden_dim))

        self.w_in = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.w_hn = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_in = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_hn = nn.Parameter(torch.Tensor(hidden_dim))

        self.reset_params()

    def reset_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, h):
        # TODO: your codes here
        r = torch.sigmoid(torch.matmul(input, self.w_ir) + self.b_ir + torch.matmul(h[0], self.w_hr) + self.b_hr)
        z = torch.sigmoid(torch.matmul(input, self.w_iz) + self.b_iz + torch.matmul(h[0], self.w_hz) + self.b_hz)
        n = torch.tanh(torch.matmul(input, self.w_in) + self.b_in + r * (torch.matmul(h[0], self.w_hn) + self.b_hn))
        h_new = (1 - z) * n + z * h[0]
        return h_new, h_new


class LSTMCell(nn.Module):
    '''A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function.

    Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_dim)`: tensor containing input features
        - **h_0** of shape `(batch, hidden_dim)`: tensor containing the initial hidden
          state for each element in the batch.
        - **c_0** of shape `(batch, hidden_dim)`: tensor containing the initial cell state
          for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

    Outputs: h_1, c_1
        - **h_1** of shape `(batch, hidden_dim)`: tensor containing the next hidden state
          for each element in the batch
        - **c_1** of shape `(batch, hidden_dim)`: tensor containing the next cell state
          for each element in the batch
    '''
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.w_ii = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.w_hi = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_ii = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_hi = nn.Parameter(torch.Tensor(hidden_dim))

        self.w_if = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.w_hf = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_if = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_hf = nn.Parameter(torch.Tensor(hidden_dim))

        self.w_ig = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.w_hg = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_ig = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_hg = nn.Parameter(torch.Tensor(hidden_dim))

        self.w_io = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.w_ho = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_io = nn.Parameter(torch.Tensor(hidden_dim))
        self.b_ho = nn.Parameter(torch.Tensor(hidden_dim))

        self.reset_params()

    def reset_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        # TODO: your codes here
        i = torch.sigmoid(torch.matmul(input, self.w_ii) + self.b_ii + torch.matmul(state[0], self.w_hi) + self.b_hi)
        f = torch.sigmoid(torch.matmul(input, self.w_if) + self.b_if + torch.matmul(state[0], self.w_hf) + self.b_hf)
        g = torch.tanh(torch.matmul(input, self.w_ig) + self.b_ig + torch.matmul(state[0], self.w_hg) + self.b_hg)
        o = torch.sigmoid(torch.matmul(input, self.w_io) + self.b_io + torch.matmul(state[0], self.w_ho) + self.b_ho)
        c = f * state[1] + i * g
        h = o * torch.tanh(c)
        return h, c
