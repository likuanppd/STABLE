import torch
from torch import nn
import math
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, features):
        """ Graph Convolutional Layer forward function
        """
        if features.data.is_sparse:
            support = torch.spmm(features, self.weight)
        else:
            support = torch.mm(features, self.weight)
        if adj.is_sparse:
            output = torch.sparse.mm(adj, support)
        else:
            output = adj @ support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_dims):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(in_feats, n_hidden))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.5))
        self.layers.append(GraphConvolution(n_hidden, out_dims))

    def forward(self, adj, features):
        h = features
        for layer in self.layers:
            if isinstance(layer, nn.Dropout) or isinstance(layer, nn.ReLU):
                h = layer(h)
            else:
                h = layer(adj, h)
        return h


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


