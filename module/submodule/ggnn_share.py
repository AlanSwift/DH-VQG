import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import adj_dense2edge_sparse
import math


class GGNN(nn.Module):
    def __init__(self, graph_hops, d_in, d_out, dropout=0.2, step=100):
        super(GGNN, self).__init__()
        self.graph_hops = graph_hops
        # hop 1
        self.msg_passing_fw = MessagePassing(d_in=d_in, d_out=d_in)
        self.msg_passing_bw = MessagePassing(d_in=d_in, d_out=d_in)
        self.gated_fusion = GatedFusion(d_in)

        self.gru_step = GRUStep(input_size=d_in, hidden_size=d_in)
        self.d_gru = d_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm([step, d_in])
        self.out_prj = nn.Linear(d_in, d_out)


    def forward(self, x, adj):
        for i in range(self.graph_hops):
            x_in = self.dropout(x)
            bw_agg_state = self.msg_passing_bw(x_in, adj)
            fw_agg_state = self.msg_passing_fw(x_in, adj.transpose(1, 2))
            agg_state = self.gated_fusion(fw_agg_state, bw_agg_state)
            x = self.gru_step(h_state=x_in, input=agg_state)
            # x = self.norm(x)
            x = (x + x_in) / math.sqrt(2)
        x = self.dropout(x)
        x = self.out_prj(x)
        x = F.relu(x)
        return x


class MessagePassing(nn.Module):
    def __init__(self, d_in, d_out):
        super(MessagePassing, self).__init__()
        # self.linears = nn.Linear(d_in, d_out)
        # self.activate = nn.ReLU()

    def forward(self, x, adj):

        # Add self-loop
        norm_ = torch.sum(adj, 2, keepdim=True) + 1
        agg_state = (torch.bmm(adj, x) + x) / norm_
        return agg_state


class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        '''GatedFusion module'''
        self.fc_z = nn.Linear(4 * hidden_size, hidden_size, bias=True)

    def forward(self, h_state, input):
        z = torch.sigmoid(self.fc_z(torch.cat([h_state, input, h_state * input, h_state - input], -1)))
        h_state = (1 - z) * h_state + z * input
        return h_state


class GRUStep(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUStep, self).__init__()
        '''GRU module'''
        self.linear_z = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        # self.linear_r = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        # self.linear_t = nn.Linear(hidden_size + input_size, hidden_size, bias=False)

    def forward(self, input, h_state):
        z = torch.sigmoid(self.linear_z(torch.cat([h_state, input], -1)))
        # r = torch.sigmoid(self.linear_r(torch.cat([h_state, input], -1)))
        # t = torch.tanh(self.linear_t(torch.cat([r * h_state, input], -1)))
        h_state = (1 - z) * h_state + z * input
        return h_state
