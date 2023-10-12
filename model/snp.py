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

class PyramidLayer(nn.Module):
    def __init__(self, dim, t_ratio, qkv_bias=True) -> None:
        super().__init__()
        self.first_layer = nn.Conv2d(dim, int(dim * t_ratio), kernel_size=1, bias=qkv_bias)
        t_q_convs = []
        for i in range(3):
            t_q_convs.append(nn.Conv2d(int(dim * t_ratio), int(dim * t_ratio), kernel_size=(2, 1),stride=(2,1), bias=qkv_bias))
        self.t_q_convs = nn.ModuleList(t_q_convs)
        self.last = nn.Conv2d(22, 12, kernel_size=1, bias=qkv_bias)


    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.first_layer(x)
        results = []
        results.append(x)
        for t_q_conv in self.t_q_convs:
            x = t_q_conv(x)
            results.append(x)
        y = torch.cat(results, dim=2).permute(0, 2, 3, 1)
        y = self.last(y).permute(0, 2, 1, 3)
        return y

class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size = 1, c_out= -1):
        super(ConvLayer, self).__init__()
        if c_out == -1:
            c_out = c_in // 2

        self.downConv = nn.Conv2d(in_channels=c_in,
                                  out_channels=c_out,
                                  kernel_size=window_size,
                                  stride=window_size)
        self.norm = nn.BatchNorm2d(c_out)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""

    def __init__(self, d_out, scale_size, node_size=307):
        super(Bottleneck_Construct, self).__init__()
        self.conv_layers = []
        up_size = 0
        for i in range(scale_size):
            cur_out_size = d_out//(2**i)
            self.conv_layers.append(ConvLayer(cur_out_size))
            up_size += cur_out_size //2
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.up = ConvLayer(up_size, c_out=d_out)
        self.up_att = nn.Sequential(*[
            nn.Conv2d(in_channels=up_size,
                      out_channels=d_out,
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(d_out),
            nn.Tanh()
        ])
        self.norm = nn.LayerNorm([d_out, node_size])
        self.d_out = d_out
        self.down = ConvLayer(d_out, c_out=d_out)
        self.down_att = nn.Sequential(*[
            nn.Conv2d(in_channels=d_out,
                      out_channels=d_out,
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(d_out),
            nn.Tanh()
        ])

    def forward(self, enc_input):
        # enc_input torch.Size([16, 12, 307, 64])
        # temp_input = self.down(enc_input).permute(0, 2, 1)
        # assert enc_input.shape[1] == self.d_out
        temp_input = enc_input
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input)

        all_inputs = torch.cat(all_inputs, dim=1)
        all_inputs_v = self.up(all_inputs)
        all_inputs_a = self.up_att(all_inputs)
        
        enc_input_v = self.down(enc_input)
        enc_input_a = self.down_att(enc_input)

        all_inputs = all_inputs_v * all_inputs_a + enc_input_v * enc_input_a
        # all_inputs = self.norm(all_inputs)
        return all_inputs

class MultiScaleModule(nn.Module):
    def __init__(self, d_enc_in, window_size, node_size=12) -> None:
        super().__init__()
        self.conv_layers = Bottleneck_Construct(
            d_enc_in, 2, node_size=node_size)

    def forward(self, enc_input):
        B, N, T, H = enc_input.shape  # torch.Size([16, 307, 12, 64])
        enc_input1 = enc_input.permute(0, 2, 1, 3)  # torch.Size([16, 12, 307, 64])
        enc_input = self.conv_layers(enc_input1).permute(0, 2, 1, 3)  # torch.Size([16, 307, 12, 64])
        return enc_input

if __name__ == '__main__':
    model = MultiScaleModule(12, 2)
    x = torch.randn((16, 307, 12, 64))
    y = model(x)
    print(x.shape)

