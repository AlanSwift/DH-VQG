import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import adj_dense2edge_sparse


class GatedGraphNN(nn.Module):
    def __init__(self, layers, d_in, d_out):
        super(GatedGraphNN, self).__init__()
        # layer 1
        self.layer1 = nn.ModuleList([GGNNLayer(graph_hops=1, d_in=d_in, d_hidden=d_in, d_out=d_out)])
        # multilayer
        self.layer = layers
        self.layer2 = nn.ModuleList()
        for i in range(self.layer - 1):
            self.layer2.extend([GGNNLayer(graph_hops=1, d_in=d_out, d_hidden=d_out, d_out=d_out)])

    def forward(self, x, adj):
        x_out = self.layer1[0](x, adj)
        for i in range(self.layer - 1):
            x_in = x_out
            x_out = self.layer2[i](x_in, adj)
        return x_out


class GGNNLayer(nn.Module):
    def __init__(self, graph_hops, d_in, d_hidden, d_out):
        super(GGNNLayer, self).__init__()
        self.graph_hops = graph_hops
        # hop 1
        self.msg_passing = MessagePassing(d_in=d_in, d_out=d_out)
        self.gated_fusion = GatedFusion(d_out)
        self.gru_step = GRUStep(input_size=d_in, hidden_size=d_out)


    def forward(self, x, adj):
        bw_agg_state = self.msg_passing(x, adj)
        fw_agg_state = self.msg_passing(x, adj.transpose(1, 2))
        agg_state = self.gated_fusion(fw_agg_state, bw_agg_state)
        x_agg = self.gru_step(x, agg_state)

        return x_agg


class MessagePassing(nn.Module):
    def __init__(self, d_in, d_out):
        super(MessagePassing, self).__init__()
        self.linears = nn.Linear(d_in, d_out)
        self.activate = nn.ReLU()

    def forward(self, x, adj):

        # Add self-loop
        norm_ = torch.sum(adj, 2, keepdim=True) + 1
        agg_state = (torch.bmm(adj, x) + x) / norm_
        agg_state = self.linears(agg_state)
        return self.activate(agg_state)


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
        self.linear_r = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_t = nn.Linear(hidden_size + input_size, hidden_size, bias=False)

    def forward(self, input, h_state):
        z = torch.sigmoid(self.linear_z(torch.cat([h_state, input], -1)))
        r = torch.sigmoid(self.linear_r(torch.cat([h_state, input], -1)))
        t = torch.tanh(self.linear_t(torch.cat([r * h_state, input], -1)))
        h_state = (1 - z) * h_state + z * t
        return h_state
