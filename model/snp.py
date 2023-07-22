import torch
from torch import nn
import numpy as np
import math

class SNPModule(nn.Module):
    def __init__(self, embed_size:int, forward_expansion: float, liner_sub_sample = 8, dropout:float = 0., *args, **kward) -> None:
        super(SNPModule, self).__init__(*args, **kward)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.snp_forward_fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.snp_forward_fc2 = nn.Linear(forward_expansion * embed_size, embed_size)
        self.rho1 = nn.Sequential(
            nn.Linear(embed_size, embed_size//liner_sub_sample),
            nn.Tanh(),
            nn.Linear(embed_size//liner_sub_sample, embed_size)
        )

        self.rho2 = nn.Sequential(
            nn.Linear(embed_size, embed_size//liner_sub_sample),
            nn.Tanh(),
            nn.Linear(embed_size//liner_sub_sample, embed_size)
        )

        self.g = nn.Hardswish()
        self.f = self.g
        self.dropout = nn.Dropout(dropout)
        self.u = nn.Parameter(torch.Tensor(1, embed_size))

    # x:torch.Size([4912, 144])  
    def forward(self, x, T = 0):
        self.zero_x = torch.zeros_like(x)
        # x.size: (b, n)
        # if x > T fire
        fire_data = torch.where(x > T, x, self.zero_x)
        # if x < T not fire
        not_fire_data = torch.where(x < T, x, self.zero_x)

        forward_fire = self.rho1(fire_data) * fire_data - self.rho2(fire_data) * self.g(fire_data)

        forward_not_fire = self.feed_forward(not_fire_data)
        forward = forward_fire + forward_not_fire
        return forward


class SNPModule1(nn.Module):
    def __init__(self, embed_size:int, forward_expansion: float, liner_sub_sample = 8, dropout:float = 0., *args, **kward) -> None:
        super(SNPModule1, self).__init__(*args, **kward)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.snp_forward_fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.snp_forward_fc2 = nn.Linear(forward_expansion * embed_size, embed_size)
        self.rho1 = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            # nn.Tanh(),
            # nn.Linear(embed_size//liner_sub_sample, embed_size)
        )

        self.rho2 = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            # nn.Tanh(),
            # nn.Linear(embed_size//liner_sub_sample, embed_size)
        )

        self.g = nn.Softmax()
        self.f = nn.Softmax()
        self.dropout = nn.Dropout(dropout)
        # self.u = nn.Parameter(torch.Tensor(1, embed_size))
        self.u = None
        self.embed_size = embed_size
        # self.init_weight()

    def init_weight(self):
        stdv = 1.0 / math.sqrt(self.embed_size)
        self.u.data.uniform_(-stdv, stdv)

    # x:torch.Size([4912, 144])  
    def forward(self, x, T = -math.inf):
        with torch.no_grad():
            if self.u is None:
                self.u = torch.zeros_like(x).mean(0)
        x = self.u + x
        self.zero_x = torch.zeros_like(x)
        nedd_fire_data = torch.where(x > T, x, self.zero_x)
        # if x < T not fire
        not_need_fire_data = torch.where(x < T, x, self.zero_x)
        fired_data = nedd_fire_data - self.g(nedd_fire_data)
        not_fired_data = not_need_fire_data
        u = fired_data + not_fired_data

        u = torch.mean(u, dim=0).reshape(1, -1)
        h =  self.f(self.u + x)

        with torch.no_grad():
            # genarate function
            self.u.data = u
        return h

class SNPAttention(nn.Module):
    def __init__(self, embed_size):
        super(SNPAttention, self).__init__()
        self.snp = SNPModule1(embed_size, 1)

    # Q,K,V:torch.size([16, 4, 307, 12, 16])
    def forward(self, Q, K, V):
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)    # torch.size([16, 4, 307, 12, 12])
        # # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        # attn = nn.Softmax(dim=-1)(scores)
        b, heads, n, hidden_dim, _ = scores.size()
        attns = []
        # batch size hidden 307, vector_length, vector_length
        for i in range(heads):
            # batch size 307, vector_length, vector_length
            # batch_size * 307, vector_length * vector_length
            score = scores[:,i,...]
            score = score.reshape([b*n, hidden_dim * hidden_dim])  # torch.Size([4912, 144])
            
            attn_i = self.snp(score)   # torch.Size([4912, 144])
            attn_i = attn_i.reshape(b,1,n, hidden_dim, hidden_dim)  # torch.Size([16, 1, 307, 12, 12])
            attns.append(attn_i)
        attn = torch.concat(attns, dim=1)
        context = torch.matmul(attn,
                               V)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context


class SNPLSTM(nn.Module):
    def __init__(self, *args, **kward) -> None:
        super().__init__(*args, **kward)

    def forward(self, x:torch.tensor)->int:
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        x : torch.tensor
            _description_

        Returns
        -------
        int
            _description_

        Raises
        ------
        TypeError
            _description_
        """
        if isinstance(x) == torch.tensor:
            raise TypeError("The type of input must be torch.tensor")
        