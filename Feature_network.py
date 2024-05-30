import torch
import torch.nn as nn
import math
import torch.nn.functional as F

EPS = 1e-30


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.44)
        if bias:
            self.bias = nn.Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight) - self.bias
        return out

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


class PowerLayer(nn.Module):
    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))


class HGLEnet(nn.Module):
    def apply_bn(self, x):

        bn_module = torch.nn.BatchNorm1d(x.size()[1]).to(self.device)
        return bn_module(x)

    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
        )

    def __init__(self,device, num_class, input_size, sampling_rate, num_T, out_graph, dropout_rate, pool, pool_step_rate):
        super(HGLEnet, self).__init__()
        self.window = [0.5, 0.2, 0.1]
        self.pool = pool
        self.channel = input_size[1]
        self.device=device
        self.Tception1 = self.temporal_learner(input_size[0], num_T, (1, int(self.window[0] * sampling_rate)),
                                               self.pool,
                                               pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T, (1, int(self.window[1] * sampling_rate)),
                                               self.pool,
                                               pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T, (1, int(self.window[2] * sampling_rate)),
                                               self.pool,
                                               pool_step_rate)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_t_ = nn.BatchNorm2d(num_T)
        self.OneXOneConv = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, 2))
        )
        size = self.get_size_temporal(input_size)
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)
        self.global_adj = nn.Parameter(torch.FloatTensor(self.channel, self.channel), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)

        self.bn = nn.BatchNorm1d(self.channel)
        self.bn_ = nn.BatchNorm1d(self.channel)

        self.entry_conv_first = GraphConvolution(in_features=size[-1], out_features=out_graph)

    # (batch_size,1,channels,features)
    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.apply_bn(out)
        adj = self.get_adj(out)
        out_all_entry = []
        layer_out_1 = self.entry_conv_first(out, adj)
        layer_out_1 = self.bn_(layer_out_1)




        out_all_entry.append(layer_out_1)
        out_all_entry = torch.cat(out_all_entry, dim=2)
        out_mean_1 = torch.mean(out_all_entry, dim=1)
        transformed_global_1 = torch.tanh(out_mean_1)
        sigmoid_scores_1 = F.softmax(torch.matmul(out_all_entry, transformed_global_1.unsqueeze(-1)), dim=-1)
        representation_1 = torch.matmul(torch.transpose(out_all_entry, 2, 1), sigmoid_scores_1)
        representation_1 = torch.transpose(representation_1, 2, 1).squeeze(1)
        output = representation_1

        return output

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_adj(self, x, self_loop=True):
        # x: b, node, feature
        adj = self.self_similarity(x)  # b, n, n
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(self.device)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def self_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s




