import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
import copy
from deeprobust.graph.utils import *
from utils import sparse_dense_mul


def get_contrastive_emb(logger, adj, features, adj_delete, lr, weight_decay, nb_epochs, beta, recover_percent=0.2):
    ft_size = features.shape[2]
    nb_nodes = features.shape[1]
    aug_features1 = features
    aug_features2 = features
    aug_adj1 = aug_random_edge(adj, adj_delete=adj_delete, recover_percent=recover_percent)  # random drop edges
    aug_adj2 = aug_random_edge(adj, adj_delete=adj_delete, recover_percent=recover_percent)  # random drop edges
    adj = normalize_adj(adj + (sp.eye(adj.shape[0]) * beta))
    aug_adj1 = normalize_adj2(aug_adj1 + (sp.eye(adj.shape[0]) * beta))
    aug_adj2 = normalize_adj2(aug_adj2 + (sp.eye(adj.shape[0]) * beta))
    sp_adj = sparse_mx_to_torch_sparse_tensor((adj))
    sp_aug_adj1 = sparse_mx_to_torch_sparse_tensor(aug_adj1)
    sp_aug_adj2 = sparse_mx_to_torch_sparse_tensor(aug_adj2)
    model = DGI(ft_size, 512, 'prelu')
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        logger.info('Using CUDA')
        model.cuda()
        features = features.cuda()
        aug_features1 = aug_features1.cuda()
        aug_features2 = aug_features2.cuda()
        sp_adj = sp_adj.cuda()
        sp_aug_adj1 = sp_aug_adj1.cuda()
        sp_aug_adj2 = sp_aug_adj2.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]

        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()

        logits = model(features, shuf_fts, aug_features1, aug_features2,
                       sp_adj, sp_aug_adj1, sp_aug_adj2,
                       True, None, None, None, aug_type='edge')
        loss = b_xent(logits, lbl)
        logger.info('Loss:[{:.4f}]'.format(loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            weights = copy.deepcopy(model.state_dict())
        else:
            cnt_wait += 1

        if cnt_wait == 20:
            logger.info('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    logger.info('Loading {}th epoch'.format(best_t))
    model.load_state_dict(weights)

    return model.embed(features, sp_adj, True, None)

def aug_random_edge(input_adj, adj_delete, recover_percent=0.2):
    percent = recover_percent
    adj_delete = sp.tril(adj_delete)
    row_idx, col_idx = adj_delete.nonzero()
    edge_num = int(len(row_idx))
    add_edge_num = int(edge_num * percent)
    print("the number of recovering edges: {:04d}" .format(add_edge_num))
    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_list = [(i, j) for i, j in zip(row_idx, col_idx)]
    edge_idx = [i for i in range(edge_num)]
    add_idx = random.sample(edge_idx, add_edge_num)

    for i in add_idx:
        aug_adj[edge_list[i][0]][edge_list[i][1]] = 1
        aug_adj[edge_list[i][1]][edge_list[i][0]] = 1


    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_adj2(adj, alpha=-0.5):
    """Symmetrically normalize adjacency matrix."""
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = torch.add(torch.eye(adj.shape[0]), adj)
    degree = adj.sum(dim=1)
    in_degree_norm = torch.pow(degree.view(1, -1), alpha).expand(adj.shape[0], adj.shape[0])
    out_degree_norm = torch.pow(degree.view(-1, 1), alpha).expand(adj.shape[0], adj.shape[0])
    adj = sparse_dense_mul(adj, in_degree_norm)
    adj = sparse_dense_mul(adj, out_degree_norm)
    if alpha > 0:
        return to_scipy((adj / (adj.sum(dim=1).reshape(adj.shape[0], -1)))).tocoo()
    else:
        return to_scipy(adj).tocoo()


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN_DGI(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

    # (features, shuf_fts, aug_features1, aug_features2,
    #  sp_adj if sparse else adj,
    #  sp_aug_adj1 if sparse else aug_adj1,
    #  sp_aug_adj2 if sparse else aug_adj2,
    #  sparse, None, None, None, aug_type=aug_type
    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2, aug_type):
        h_0 = self.gcn(seq1, adj, sparse)
        if aug_type == 'edge':

            h_1 = self.gcn(seq1, aug_adj1, sparse)
            h_3 = self.gcn(seq1, aug_adj2, sparse)

        elif aug_type == 'mask':

            h_1 = self.gcn(seq3, adj, sparse)
            h_3 = self.gcn(seq4, adj, sparse)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = self.gcn(seq3, aug_adj1, sparse)
            h_3 = self.gcn(seq4, aug_adj2, sparse)

        else:
            assert False

        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        c_3 = self.read(h_3, msk)
        c_3 = self.sigm(c_3)

        h_2 = self.gcn(seq2, adj, sparse)

        ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)

        ret = ret1 + ret2
        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class GCN_DGI(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN_DGI, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        tmp = self.f_k(h_pl, c_x)
        sc_1 = torch.squeeze(tmp, 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits