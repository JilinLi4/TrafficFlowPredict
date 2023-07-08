import math
import torch
from torch import nn


class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, *args, **kwars) -> None:
        super(CustomLSTM, self).__init__()
        self.input_sz   = input_sz
        self.hidden_sz  = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weight()

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states = None):
        """
        Parameters
        ----------
        x : _type_ torch.Tensor
            Assumes x is of shape (batch, sequence, feature)
        init_states : _type_, optional
            _description_, by default None
        """

        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_sz).to(x.device)
            c_t = torch.zeros(bs, self.hidden_sz).to(x.device)
        else:
            h_t , ct = init_states

        HS = self.hidden_sz
        for t in range(seq_sz):
            x_t = x[:, t, :]
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t = torch.sigmoid(gates[:,        : HS])      # input1
            f_t = torch.sigmoid(gates[:, HS     : HS * 2])  # forget
            g_t = torch.sigmoid(gates[:, HS * 2 : HS * 3])  # input2
            o_t = torch.sigmoid(gates[:, HS * 3 : HS * 4])  # output
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


if __name__ == '__main__':
    x = torch.randn(16, 12, 1).cuda()
    lstm = CustomLSTM(1, 1).cuda()
    y,_ = lstm(x)
    print(y.size())