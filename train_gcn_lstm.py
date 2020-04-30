import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse


def aaa2id(aa):
    table = dict(ALA=0, ARG=1, ASN=2, ASP=3, CYS=4, GLN=5, GLU=6, GLY=7, HIS=8,
                 ILE=9, LEU=10, LYS=11, MET=12, PHE=13, PRO=14, SER=15, THR=16,
                 TRP=17, TYR=18, VAL=19)
    return table[aa.upper()]


def a2id(aa):
    table = dict(A=0, R=1, N=2, D=3, C=4, E=5, Q=6, G=7, H=8,
                 I=9, L=10, K=11, M=12, F=13, P=14, S=15, T=16,
                 W=17, Y=18, V=19, X=19)
    return table[aa.upper()]


def id2a(aid):
    table = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
             "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
             "X"]
    return table[aid]


class GCNDataset(Dataset):

    def __init__(self, n, max_buf_size, df_path, seq_len_range=(128, 512), seed=0, verbose=False):
        self.rn = np.random.RandomState(seed)
        self.n = n
        self.verbose = verbose
        self.df_path = df_path
        self.idxs = list()
        self.f = list()
        self.y = list()
        self.seq = list()
        self.gt_seq = list()
        self.gt_idxs = list()
        self.build(max_buf_size, seq_len_range)

    def build(self, max_buf_size, seq_len_range):
        if self.df_path:
            df = pd.read_hdf(self.df_path, "df").query("len >= {} and len <= {} and standard".format(seq_len_range[0],
                                                                                                     seq_len_range[1]))
            print("using {} ({} sequences)".format(self.df_path, len(df)))
            if self.n > 0:
                df = df.sample(self.n)
        for i in range(self.n):
            if self.df_path:
                my_seq_len = df["len"].iloc[i]
            else:
                my_seq_len = self.rn.randint(seq_len_range[0], seq_len_range[1])
            my_seq_idxs = np.arange(my_seq_len)
            if self.verbose:
                print("my_seq_idxs", my_seq_idxs.shape)
            my_dummy_idxs = self.rn.permutation(np.arange(seq_len_range[1], 2 * max_buf_size))[
                            0:(max_buf_size - my_seq_len)]
            if self.verbose:
                print("my_dummy_idxs", my_dummy_idxs.shape)
            my_rand_idxs = self.rn.permutation(max_buf_size)
            if self.verbose:
                print("my_rand_idxs", my_rand_idxs.shape)
            my_idxs = np.concatenate((my_seq_idxs, my_dummy_idxs))[my_rand_idxs]
            if self.verbose:
                print("my_idxs", my_idxs.shape)
            my_y = 1 * (my_idxs < my_seq_len)
            if self.df_path:
                my_seq = np.array([a2id(c) for c in df["seq"].iloc[i]])
            else:
                my_seq = self.rn.randint(0, 20, my_seq_len)
            my_seq = np.concatenate((my_seq, self.rn.randint(0, 20, max_buf_size-my_seq_len)))
            my_seq = my_seq[my_rand_idxs]
            my_feats = 0.01 * abs(self.rn.randn(len(my_seq), 20))
            my_gt_idxs = np.argsort(my_idxs)[0:my_seq_len]
            my_gt_seq = my_seq[my_gt_idxs]
            # print("GT", df["seq"].iloc[i])
            # print("  ", "".join([id2a(j) for j in my_gt_seq]))
            for j, s in enumerate(my_seq):
                my_feats[j, s] = 1 - np.sum(my_feats[j, :]) + my_feats[j, s]
            self.idxs.append(my_idxs)
            self.f.append(my_feats)
            self.y.append(my_y)
            self.seq.append(my_seq)
            self.gt_seq.append(my_gt_seq)
            self.gt_idxs.append(my_gt_idxs)

    def onehot(self, x):
        z = np.zeros((len(x), 20))
        for i, c in enumerate(x):
            z[i, c] = 1
        return z

    def make_a_matrix(self, idxs, self_loop=True):
        mat = (np.repeat(idxs.reshape(-1, 1), idxs.shape[0], 1) - np.repeat(idxs.reshape(1, -1), idxs.shape[0], 0))
        mat = 1.0 * (abs(mat) == 1)
        if self_loop:
            mat = mat + np.eye(len(idxs))
        return mat

    def make_d_matrix(self, a_mat):
        mat = np.eye(a_mat.shape[0]) * np.sum(a_mat, axis=1) ** (-0.5)
        return mat

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        my_gt_seq = self.gt_seq[idx]
        my_idxs = self.idxs[idx]
        x = self.onehot(my_gt_seq)
        y = self.y[idx]
        f = self.f[idx]
        a_mat = self.make_a_matrix(my_idxs)
        d_mat = self.make_d_matrix(a_mat)
        gt_idxs = self.gt_idxs[idx]

        return x, y, f, a_mat, d_mat, gt_idxs


class GCN(nn.Module):

    def __init__(self, n_features, n_out, n_h1=0, n_h2=0, bias=False):
        super(GCN, self).__init__()
        if n_h1 == 0:
            self.W1 = nn.Linear(n_features, n_out, bias=bias)
            self.n_layers = 1
        else:
            self.W1 = nn.Linear(n_features, n_h1, bias=bias)
            if n_h2 == 0:
                self.W2 = nn.Linear(n_h1, n_out, bias=bias)
                self.n_layers = 2
            else:
                self.W2 = nn.Linear(n_h1, n_h2, bias=bias)
                self.W3 = nn.Linear(n_h2, n_out, bias=bias)
                self.n_layers = 3

    def forward(self, A, D, F):
        d = torch.mm(D, A)
        d = torch.mm(d, D)
        # layer 1
        x = torch.mm(d, F)
        x = self.W1(x)
        # layer 2
        if self.n_layers > 1:
            x = torch.mm(d, nn.ReLU()(x))
            x = self.W2(x)
        # layer 3
        if self.n_layers > 2:
            x = torch.mm(d, nn.ReLU()(x))
            x = self.W3(x)
        return x


class Generator(nn.Module):

    def __init__(self, n_feat=20, n_node_embed=64, n_graph_embed=128, n_seq_embed=16, n_seq_alphabets=20,
                 n_graph_layers=1, gpu=False):
        super(Generator, self).__init__()
        self.feat2embed = nn.Linear(n_feat, n_node_embed)
        if n_graph_layers == 1:
            self.gcn1 = GCN(n_node_embed, n_node_embed, 0, 0)
            self.gcn2 = GCN(n_node_embed, n_node_embed, 0, 0)
        elif n_graph_layers == 2:
            self.gcn1 = GCN(n_node_embed, n_node_embed, n_node_embed, 0)
            self.gcn2 = GCN(n_node_embed, n_node_embed, n_node_embed, 0)
        else:
            self.gcn1 = GCN(n_node_embed, n_node_embed, n_node_embed, n_node_embed)
            self.gcn2 = GCN(n_node_embed, n_node_embed, n_node_embed, n_node_embed)
        self.gcn2graph = nn.Linear(n_node_embed, n_graph_embed)
        self.gcn2gating = nn.Linear(n_node_embed, n_graph_embed)
        self.graph2addedge = nn.Linear(n_node_embed + n_seq_embed + n_node_embed + n_graph_embed, 1)
        self.seq2embed = nn.Linear(n_seq_alphabets, n_seq_embed)
        self.gpu = gpu

    def nodes2graph(self, h_nodes):
        h_graph = self.gcn2graph(h_nodes)
        g_graph = self.gcn2gating(h_nodes)
        return torch.sum(h_graph * nn.Sigmoid()(g_graph), dim=0)

    def forward(self, x_tensor, f_tensor, a_tensor, d_tensor):
        n = int(x_tensor.size(0))
        seq_embed = self.seq2embed(x_tensor)
        in_mask = torch.ones(f_tensor.size(0), 1)
        scores = torch.zeros(n, f_tensor.size(0))
        if self.gpu:
            in_mask = in_mask.cuda()
            # out_mask.cuda()
            scores = scores.cuda()
        f_nodes_in = self.feat2embed(f_tensor)
        h_nodes_in = self.gcn1(a_tensor, d_tensor, nn.ReLU()(f_nodes_in))
        idxs = list()
        h_nodes = torch.zeros(1, h_nodes_in.size(1))
        h_graph = torch.zeros(1, h_nodes_in.size(1) * 2)
        if self.gpu:
            h_nodes = h_nodes.cuda()
            h_graph = h_graph.cuda()
        for k in range(n):
            z = torch.cat(
                (h_nodes_in,
                 h_nodes[-1, :].view(1, -1).repeat(h_nodes_in.size(0), 1),
                 seq_embed[k, :].view(1, -1).repeat(h_nodes_in.size(0), 1),
                 h_graph.view(1, -1).repeat(h_nodes_in.size(0), 1),
                 ), dim=1)
            z = self.graph2addedge(nn.ReLU()(z) * in_mask)
            scores[k, :] = z.view(-1)
            idx_addnode = torch.argmax(scores[k, :], dim=0)
            in_mask = in_mask.clone()
            in_mask[idx_addnode, 0] = 0
            idxs.append(int(idx_addnode))
            # update graph
            a_new = torch.eye(k + 2)
            a_new = torch.eye(k + 1) + a_new[1:, 0:-1].view(k + 1, k + 1) + a_new[0:-1, 1:].view(k + 1, k + 1)
            d_new = torch.eye(k + 1) * (torch.sum(a_new, dim=1, keepdim=True) ** (-0.5))
            if self.gpu:
                a_new = a_new.cuda()
                d_new = d_new.cuda()
            # noinspection PyCallingNonCallable
            h_nodes = self.gcn2(a_new,
                                 d_new,
                                 f_nodes_in[idxs, :].view(len(idxs), -1))
            h_graph = self.nodes2graph(h_nodes)
        return scores, idxs


# noinspection PyCallingNonCallable
class GeneratorLSTM(nn.Module):

    def __init__(self, n_feat=20, n_node_embed=64, n_graph_embed=128, n_seq_embed=16, n_seq_alphabets=20,
                 n_graph_layers=1, gpu=False):
        super(GeneratorLSTM, self).__init__()
        self.n_graph_embed = n_graph_embed
        self.feat2embed = nn.Linear(n_feat, n_node_embed)
        if n_graph_layers == 1:
            self.gcn1 = GCN(n_node_embed, n_node_embed, 0, 0)
        elif n_graph_layers == 2:
            self.gcn1 = GCN(n_node_embed, n_node_embed, n_node_embed, 0)
        else:
            self.gcn1 = GCN(n_node_embed, n_node_embed, n_node_embed, n_node_embed)
        self.lstm = nn.LSTM(n_seq_embed, n_graph_embed)
        self.node2addedge = nn.Linear(n_node_embed, n_graph_embed)
        self.graph2addedge = nn.Linear(n_graph_embed, 1)
        self.nodes2gating = nn.Linear(n_node_embed, n_graph_embed*2)
        self.nodes2graph = nn.Linear(n_node_embed, n_graph_embed*2)
        self.seq2embed = nn.Linear(n_seq_alphabets, n_seq_embed)
        self.gpu = gpu

    def get_graph(self, h_nodes):
        h_graph = self.nodes2graph(h_nodes)
        h_graph = torch.sum(nn.Sigmoid()(self.nodes2gating(h_nodes)) * h_graph, dim=0)
        return h_graph

    def forward(self, x_tensor, f_tensor, a_tensor, d_tensor):
        n = int(x_tensor.size(0))
        seq_embed = self.seq2embed(x_tensor)
        scores = torch.zeros(n, f_tensor.size(0))
        if self.gpu:
            scores = scores.cuda()
        f_nodes_in = self.feat2embed(f_tensor)
        h_nodes_in = self.gcn1(a_tensor, d_tensor, nn.ReLU()(f_nodes_in))
        h_graph = self.get_graph(h_nodes_in)
        idxs = list()
        for k in range(n):
            if k == 0:
                h = h_graph[0:self.n_graph_embed].view(1, 1, -1)
                c = h_graph[self.n_graph_embed:].view(1, 1, -1)
            h_graph, (h, c) = self.lstm(seq_embed[k, :].view(1, 1, -1), (h, c))
            # z = torch.cat(
            #     (h_nodes_in,
            #      h_graph.view(1, -1).repeat(h_nodes_in.size(0), 1),
            #      ), dim=1)
            z = self.node2addedge(h_nodes_in) + h_graph.view(1, -1).repeat(h_nodes_in.size(0), 1)
            # z = self.graph2addedge(z)
            z = self.graph2addedge(nn.ReLU()(z))
            scores[k, :] = z.view(-1)
            idx_addnode = torch.argmax(scores[k, :], dim=0)
            idxs.append(int(idx_addnode))
        return scores, idxs


class LSTMSeq(nn.Module):

    def __init__(self, gcn_params, n_input, n_embed, n_hidden, n_output, gcn_0th_only=False):
        super(LSTMSeq, self).__init__()
        self.gcn = GCN(n_features=gcn_params["n_features"], n_h1=gcn_params["n_h1"], n_h2=gcn_params["n_h2"],
                       n_out=gcn_params["n_out"],
                       bias=gcn_params["bias"])
        self.gcn2embed = nn.Linear(gcn_params["n_out"] * gcn_params["max_buf_size"], n_embed)
        self.seq2embed = nn.Linear(n_input, n_embed)
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lstm = nn.LSTM(n_embed, n_hidden)
        self.hidden2tag = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        self.gcn_0th_only = gcn_0th_only

    def forward(self, a, d, f, sequence):
        gcn_out = self.relu(self.gcn2embed(self.relu(self.gcn(a, d, f).view(1, -1))))
        #         print("gcn_out", gcn_out.size())
        x1 = self.seq2embed(sequence).view(-1, 1, self.n_embed)
        x2 = gcn_out.view(1, 1, self.n_embed)
        #         print("x1", x1.size(), "x2", x2.size())
        if self.gcn_0th_only:
            x1[0, :, :] = x1[0, :, :] + x2
            embed = self.relu(x1)
        else:
            embed = self.relu(x1 + x2)
        lstm_out, _ = self.lstm(embed)
        tag_out = self.hidden2tag(lstm_out.view(-1, self.n_hidden))
        return tag_out


def train(model, dataset, val_dataset=None, n_epoch=1, lr=0.1, print_every=100, log_every=100,
          gpu=False, verbose=False):
    print("====================== train ======================")
    log = dict(train=list(), val=list())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for i in range(n_epoch):
        for j in range(len(dataset)):
            model.zero_grad()
            x, _, f, a_mat, d_mat, gt_idxs = dataset[j]
            n = len(gt_idxs)
            if verbose:
                print(j, list(gt_idxs), n)
            x_tensor = torch.from_numpy(x).float()
            f_tensor = torch.from_numpy(f).float()
            a_tensor = torch.from_numpy(a_mat).float()
            d_tensor = torch.from_numpy(d_mat).float()
            g_tensor = torch.from_numpy(gt_idxs).long()
            if gpu:
                x_tensor = x_tensor.cuda()
                f_tensor = f_tensor.cuda()
                a_tensor = a_tensor.cuda()
                d_tensor = d_tensor.cuda()
                g_tensor = g_tensor.cuda()
            scores, _ = model(x_tensor, f_tensor, a_tensor, d_tensor)
            loss_f = loss_fn(scores, g_tensor.long())
            loss_r = loss_fn(scores, torch.flip(g_tensor, (0, )).long())
            loss = torch.min(loss_f, loss_r)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if (j+1) % log_every == 0:
                    log["train"].append(dict(epoch=i+1, iter=j+1, loss=loss.item()))
                if (j+1) % print_every == 0:
                    print("epoch {} loss {:.4f}".format(i, loss.data))
        if val_dataset is not None:
            with torch.no_grad():
                val_acc = val(model, val_dataset, gpu, verbose=False)
                log["val"].append(dict(epoch=i, acc=val_acc))
        print("=======================================================================================================")
    return model, log


def val(model, dataset, gpu=False, verbose=False):
    acc_all = 0
    model.eval()
    with torch.no_grad():
        for j in range(len(dataset)):
            model.zero_grad()
            x, _, f, a_mat, d_mat, gt_idxs = dataset[j]
            n = len(gt_idxs)
            x_tensor = torch.from_numpy(x).float()
            f_tensor = torch.from_numpy(f).float()
            a_tensor = torch.from_numpy(a_mat).float()
            d_tensor = torch.from_numpy(d_mat).float()
            g_tensor = torch.from_numpy(gt_idxs).long()
            if gpu:
                x_tensor = x_tensor.cuda()
                f_tensor = f_tensor.cuda()
                a_tensor = a_tensor.cuda()
                d_tensor = d_tensor.cuda()
                g_tensor = g_tensor.cuda()
            _, idxs = model(x_tensor, f_tensor, a_tensor, d_tensor)
            acc_f = np.sum(1*(np.array(idxs) == gt_idxs))
            acc_r = np.sum(1 * (np.array(idxs) == gt_idxs[::-1]))
            acc = np.maximum(acc_f, acc_r)
            acc = acc / float(len(gt_idxs))
            acc_all = acc_all + acc
            if verbose:
                print("GT", "".join([id2a(c) for c in list(torch.argmax(f_tensor[gt_idxs, :], dim=1))]))
                print("PR", "".join([id2a(c) for c in list(torch.argmax(f_tensor[idxs, :], dim=1))]))
                print("GT", list(gt_idxs), n)
                print("PR", list(idxs), n)
                print(j, acc)
    acc_all = acc_all / len(dataset)
    print("overall acc:", acc_all)
    return acc_all
            

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_train", "-n", type=int, default=10, help="Number of training samples")
    p.add_argument("--n_val", type=int, default=10, help="Number of val samples")
    p.add_argument("--n_epoch", "-e", type=int, default=1, help="Number of training epoch")
    p.add_argument("--n_graph_layers", type=int, default=1, help="Number of layers in GCN")
    p.add_argument("--max_n_seq", "-l", type=int, default=10, help="Max sequence length")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--gpu", "-g", action="store_true", help="Use GPU")
    p.add_argument("--df", type=str, default=None, help="Path to sequence database df")
    p.add_argument("--df_val", type=str, default=None, help="Path to sequence database df for validation")
    p.add_argument("--min_len", type=int, default=4, help="Minium sequence length")
    p.add_argument("--max_len", type=int, default=8, help="Maximum sequence length")
    p.add_argument("--seed", type=int, default=2020, help="Random seed for NumPy and Pandas")
    p.add_argument("--save", "-s", type=str, default=None, help="Path and file name for saving trained model")
    p.add_argument("--log", type=str, default=None, help="Path and file name for saving training log")
    p.add_argument("--verbose", "-v", action="store_true", help="Be verbose")
    return p.parse_args()

    
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    train_dataset = GCNDataset(n=args.n_train, max_buf_size=args.max_n_seq, df_path=args.df,
                               seq_len_range=(args.min_len, args.max_len), seed=args.seed, verbose=args.verbose)
    val_dataset = GCNDataset(n=args.n_val, max_buf_size=args.max_n_seq, df_path=args.df_val,
                             seq_len_range=(args.min_len, args.max_len), seed=args.seed+1, verbose=args.verbose)
    # model = Generator(n_graph_layers=args.n_graph_layers)
    model = GeneratorLSTM(n_graph_layers=args.n_graph_layers, gpu=args.gpu)
    if args.gpu:
        model = model.cuda()
        print(model)
    model, json_log = train(model, train_dataset, val_dataset=val_dataset, n_epoch=args.n_epoch, lr=args.lr,
                            gpu=args.gpu, verbose=args.verbose)
    if args.save:
        torch.save(model.state_dict(), args.save)
    if args.log:
        with open(args.log, "w") as f:
            f.write(json.dumps(json_log))
    print("test on training set")
    val(model, train_dataset, gpu=args.gpu)
    print("test on validation set")
    val(model, val_dataset, gpu=args.gpu, verbose=True)


if __name__ == "__main__":
    main()
