import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from module.submodule.ggnn_share import GRUStep
"""
1, 1, 0
0, 0, 0
1, 0, 1
"""


class CrossGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc_gate1 = nn.Linear(d_model, d_model, bias=False)
        self.fc_gate2 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x1, x2):
        g1 = torch.sigmoid(self.fc_gate1(x1))
        x2_ = g1 * x2
        g2 = torch.sigmoid(self.fc_gate2(x2))
        x1_ = g2 * x1
        return x1_, x2_


class GCNSpatialV1(nn.Module):
    def __init__(self, d_in, d_out, d_answer):
        super(GCNSpatialV1, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.fc = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(3)])
        self.fc_gate = nn.ModuleList([nn.Linear(d_in + d_answer, d_in) for _ in range(3)])
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.cross_gate = CrossGate(d_out)

    def forward(self, x, adj, answer):
        fw = self.message_passing(x, adj, self.fc[0], self.fc_gate[0], answer, need_sample=True)
        bw = self.message_passing(x, adj.transpose(1, 2), self.fc[1], self.fc_gate[1], answer, need_sample=True)
        adj_mat = torch.eye(adj.size(1)).type_as(adj).unsqueeze(0).expand(x.shape[0], -1, -1)
        slf = self.message_passing(x, adj_mat, self.fc[2], self.fc_gate[2], answer, need_sample=False)
        ret = fw + bw + slf
        return self.activate(ret)

    def message_passing(self, x, adj, fc, fc_gate, answer, need_sample=True):
        # g = torch.sigmoid(fc_gate(torch.cat((x, answer.unsqueeze(1).expand(-1, x.shape[1], -1)), dim=-1)))
        adj_sampled = adj

        # D = torch.sum(adj_sampled, -1)  # [b, 1000]
        # mask = D == 0
        #
        # D = 1. / D
        # D.masked_fill_(mask=mask, value=0)
        #
        # D_inv = torch.diag_embed(D)
        # w = torch.matmul(D_inv, adj_sampled)

        x = torch.bmm(adj_sampled, x)
        return fc(x)

    def samples(self, adj, sample_num, shuffle=False):
        adj_ori = adj.clone()
        adj_ret = torch.zeros_like(adj).fill_(-1e8).to(adj.device)

        _, idx = torch.sort(adj_ori, descending=True)

        _[:, :, :15] = F.dropout(_[:, :, :15], (15 - sample_num) / 15, self.training)

        ret = torch.zeros_like(adj_ori).fill_(-1e8).to(adj.device)
        ret.scatter_(dim=2, index=idx, src=_)

        # for i in range(adj.shape[0]):
        #     for j in range(adj.shape[1]):
        #         nei_nums = torch.sum(adj[i, j, :])
        #         if nei_nums.item() <= sample_num:
        #             continue
        #         line = adj_ret[i, j, :].nonzero(as_tuple=False)
        #         line = line.view(-1)
        #         sampled = np.random.choice(line.detach().cpu().numpy(), sample_num, replace=False)
        #         adj_ret[i, j, sampled] = adj_ori[i, j, sampled]
        return ret


class GCNEncoderSpatialV1(nn.Module):
    def __init__(self, d_in, d_out, d_answer, dropout=0.5, layer=2, step=50, need_align=True):
        super(GCNEncoderSpatialV1, self).__init__()
        self.gcn_layer = layer
        layers = []
        norm_layers = []
        for i in range(self.gcn_layer):
            layers.append(GCNSpatialV1(d_in=d_in, d_out=d_in, d_answer=d_answer))
            norm_layers.append(nn.LayerNorm([step, d_in]))
        self.layers = nn.ModuleList(layers)
        self.norm_layers = nn.ModuleList(norm_layers)
        self.dropout = nn.Dropout(dropout)
        self.need_align = need_align
        if self.need_align:
            self.out_linear = nn.Linear(d_in, d_out)
            self.activate = nn.ReLU()

    def forward(self, graph, adj, answer):

        x = graph
        for layer, norm in zip(self.layers, self.norm_layers):
            res = x
            x = layer(x, adj, answer)
            x = (res + x) / math.sqrt(2)
            x = norm(x)
            x = self.dropout(x)
        if self.need_align:
            x = self.out_linear(x)
            x = self.activate(x)
        return x


class GCNEncoderSpatialV2(nn.Module):
    def __init__(self, d_in, d_out, d_answer, dropout=0.5, layer=2, step=50, need_align=True):
        super(GCNEncoderSpatialV2, self).__init__()
        self.gcn_layer = layer
        layers = []
        norm_layers = []
        for i in range(self.gcn_layer):
            layers.append(GCNSpatialV1(d_in=d_in, d_out=d_in, d_answer=d_answer))
            norm_layers.append(nn.LayerNorm([step, d_in]))
        self.layers = nn.ModuleList(layers)
        self.norm_layers = nn.ModuleList(norm_layers)
        self.dropout = nn.Dropout(dropout)
        self.need_align = need_align
        self.gru = GRUStep(hidden_size=d_in, input_size=d_in)
        if self.need_align:
            self.out_linear = nn.Linear(d_in, d_out)
            self.activate = nn.ReLU()

    def forward(self, graph, adj, answer):

        x = graph
        for layer, norm in zip(self.layers, self.norm_layers):
            res = x
            x = layer(x, adj, answer)
            x = self.gru(h_state=res, input=x)
            # x = (res + x) / math.sqrt(2)
            x = norm(x)
            x = self.dropout(x)
        if self.need_align:
            x = self.out_linear(x)
            x = self.activate(x)
        return x


class GCNSpatial(nn.Module):
    def __init__(self, d_in, d_out):
        super(GCNSpatial, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.fc = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(3)])
        self.fc_gate = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(3)])
        self.activate = nn.ReLU()

    def forward(self, x, adj):
        fw = self.message_passing(x, adj, self.fc[0], self.fc_gate[0])
        bw = self.message_passing(x, adj.transpose(1, 2), self.fc[1], self.fc_gate[1])
        adj_mat = torch.eye(adj.size(1)).type_as(adj).unsqueeze(0).expand(x.shape[0], -1, -1)
        slf = self.message_passing(x, adj_mat, self.fc[2], self.fc_gate[2])
        ret = fw + bw + slf
        return self.activate(ret)

    def message_passing(self, x, adj, fc, fc_gate):
        x = fc(x)
        # g = torch.sigmoid(fc_gate(x))
        x = torch.bmm(adj, x)
        return x

class GCN(nn.Module):
    def __init__(self, d_in, d_out):
        super(GCN, self).__init__()
        self.fc = nn.Linear(d_in, d_out, bias=True)

    def forward(self, x, adj):
        n_node = x.shape[1]
        b = x.shape[0]
        #A=adj
        A = (adj + adj.transpose(-1, -2)) / 2.0
        A = A + torch.eye(n_node).cuda().float()
        A = A.float()
        #A = ((adj + adj.transpose(-1, -2)) > 0).float() + torch.eye(n_node).cuda().float()
        D = torch.sum(A, -1) # [b, 1000]
        mask = D == 0

        D = 1. / torch.sqrt(D)
        D.masked_fill_(mask=mask, value=0)

        D_inv = torch.diag_embed(D)

        # D = D.view(b, n_node, 1).expand(-1, -1, n_node)
        # D = 1. / torch.sqrt(D)
        #
        # D_inv = torch.eye(n_node).unsqueeze(0).expand(b, n_node, n_node).cuda() * D

        w = torch.matmul(D_inv, A)
        w = torch.matmul(w, D_inv)
        x = torch.matmul(w, x)
        x = self.fc(x)
        x = nn.ReLU()(x)

        return x


class GCNEncoder(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.5, layer=2, step=50, d_answer=None, need_align=True):
        super(GCNEncoder, self).__init__()
        self.gcn_layer = layer
        layers = []
        norm_layers = []
        for i in range(self.gcn_layer):
            layers.append(GCN(d_in=d_in, d_out=d_in))
            norm_layers.append(nn.LayerNorm([step, d_in]))
        self.layers = nn.ModuleList(layers)
        self.norm_layers = nn.ModuleList(norm_layers)
        self.dropout = nn.Dropout(dropout)
        self.need_align = need_align
        if self.need_align:
            self.out_linear = nn.Linear(d_in, d_out)
            self.activate = nn.ReLU()

    def forward(self, graph, adj, answer_hint=None):

        x = graph
        for layer, norm in zip(self.layers, self.norm_layers):
            res = x
            x = layer(x, adj)
            x = (res + x) / math.sqrt(2)
            x = norm(x)
            x = self.dropout(x)
        if self.need_align:
            x = self.out_linear(x)
            x = self.activate(x)
        return x


class GCNEncoderSpatial(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.5, layer=2, step=50, need_align=True):
        super(GCNEncoderSpatial, self).__init__()
        self.gcn_layer = layer
        layers = []
        norm_layers = []
        for i in range(self.gcn_layer):
            layers.append(GCNSpatial(d_in=d_in, d_out=d_in))
            norm_layers.append(nn.LayerNorm([step, d_in]))
        self.layers = nn.ModuleList(layers)
        self.norm_layers = nn.ModuleList(norm_layers)
        self.dropout = nn.Dropout(dropout)
        self.need_align = need_align
        if self.need_align:
            self.out_linear = nn.Linear(d_in, d_out)
            self.activate = nn.ReLU()

    def forward(self, graph, adj):

        x = graph
        for layer, norm in zip(self.layers, self.norm_layers):
            res = x
            x = layer(x, adj)
            x = (res + x) / math.sqrt(2)
            x = norm(x)
            x = self.dropout(x)
        if self.need_align:
            x = self.out_linear(x)
            x = self.activate(x)
        return x

