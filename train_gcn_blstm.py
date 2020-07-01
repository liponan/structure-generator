import os
import time
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
                df = df.sample(self.n, random_state=self.rn)
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

    def __init__(self, n_features, n_out, n_hidden=None, n_layers=1, bias=False, device=None):
        super(GCN, self).__init__()
        self.n_layers = n_layers
        print("GCN layers: {}".format(self.n_layers))
        if n_hidden is None:
            n_hidden = n_out
        if n_layers == 1:
            self.W_out = nn.Linear(n_features, n_out, bias=bias)
            if device is not None:
                self.W_out = self.W_out.to(device)
        else:
            self.W_hidden = list()
            self.W_hidden.append(nn.Linear(n_features, n_hidden, bias=bias))
            for i in range(n_layers-2):
                self.W_hidden.append(nn.Linear(n_hidden, n_hidden, bias=bias))
            if device is not None:
                for i in range(len(self.W_hidden)):
                    self.W_hidden[i] = self.W_hidden[i].to(device)
            self.W_out = nn.Linear(n_hidden, n_out, bias=bias)
            if device is not None:
                self.W_out = self.W_out.to(device)
        self.relu = nn.ReLU()

    def forward(self, mat_a, mat_d, mat_f):
        mat_d = torch.mm(torch.mm(mat_d, mat_a), mat_d)
        x = mat_f
        # hidden layers
        for i in range(self.n_layers-1):
            #print("layer {}".format(i))
            x = torch.mm(mat_d, x)
            x = self.relu(self.W_hidden[i](x))
        # output layer
            x = torch.mm(mat_d, x)
            x = self.W_out(x)
        return x


# noinspection PyCallingNonCallable
class GeneratorLSTM(nn.Module):

    def __init__(self, n_feat=20, n_node_embed=64, n_lstm_hidden=128, n_graph_embed=None, n_seq_embed=32,
                 n_seq_alphabets=20, n_graph_layers=1, n_lstm_layers=1, bidirectional_lstm=False, graph_to_lstm=False,
                 device="cpu"):
        super(GeneratorLSTM, self).__init__()
        self.graph_to_lstm = graph_to_lstm
        self.n_lstm_hidden = n_lstm_hidden
        self.n_lstm_layers = n_lstm_layers
        self.n_seq_embed = n_seq_embed
        if bidirectional_lstm:
            self.n_lstm_directions = 2
        else:
            self.n_lstm_directions = 1
        self.feat2embed = nn.Linear(n_feat, n_node_embed)
        self.gcn = GCN(n_node_embed, n_node_embed, n_node_embed, n_graph_layers, device=device, bias=True)
        self.lstm = nn.LSTM(n_seq_embed, n_lstm_hidden, num_layers=n_lstm_layers, bidirectional=bidirectional_lstm)
        self.node2addedge = nn.Linear(n_node_embed, self.n_lstm_directions*n_lstm_hidden)
        self.graph2addedge = nn.Linear(self.n_lstm_directions*n_lstm_hidden, 1)
        if self.graph_to_lstm:
            if n_graph_embed is None:
                self.n_graph_embed = 2 * n_lstm_hidden * n_lstm_layers * self.n_lstm_directions
            self.nodes2gating = nn.Linear(n_node_embed, self.n_graph_embed)
            self.nodes2graph = nn.Linear(n_node_embed, self.n_graph_embed)
        self.seq2embed = nn.Linear(n_seq_alphabets, n_seq_embed)
        self.device = device

    def get_graph(self, h_nodes):
        h_graph = self.nodes2graph(h_nodes)
        h_graph = torch.sum(nn.Sigmoid()(self.nodes2gating(h_nodes)) * h_graph, dim=0)
        return h_graph

    def forward(self, x_tensor, f_tensor, a_tensor, d_tensor):
        n = int(x_tensor.size(0))
        m = int(f_tensor.size(0))
        seq_embed = self.seq2embed(x_tensor)
        f_nodes_in = self.feat2embed(f_tensor)
        h_nodes_in = self.gcn(a_tensor, d_tensor, nn.ReLU()(f_nodes_in))
        if self.graph_to_lstm:
            h_graph = self.get_graph(h_nodes_in).view(2, -1)
            h = h_graph[0, :].view(-1, 1, self.n_lstm_hidden).to(self.device)
            c = h_graph[1, :].view(-1, 1, self.n_lstm_hidden).to(self.device)
        else:
            h = torch.zeros(self.n_lstm_layers * self.n_lstm_directions, 1, self.n_lstm_hidden).to(self.device)
            c = torch.zeros(self.n_lstm_layers * self.n_lstm_directions, 1, self.n_lstm_hidden).to(self.device)
        lstm_out, _ = self.lstm(seq_embed.view(-1, 1, self.n_seq_embed), (h, c))
        scores = self.node2addedge(h_nodes_in).view(1, m, -1).repeat(n, 1, 1)
        scores += lstm_out.view(n, 1, -1).repeat(1, m, 1)
        scores = self.graph2addedge(nn.ReLU()(scores.view(n*m, -1))).view(n, m)
        idxs = torch.argmax(scores, dim=1)
        return scores, idxs


def train(model, dataset, val_dataset=None, n_epoch=1, lr=0.1, print_every=100, log_every=100, val_every=2000,
          device=None, verbose=False):
    print("====================== train ======================")
    t0 = time.time()
    log = dict(train=list(), val=list(), val_seen=list())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for i in range(n_epoch):
        t1 = time.time()
        for j in range(len(dataset)):
            model.train()
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
            if device is not None:
                x_tensor = x_tensor.to(device)
                f_tensor = f_tensor.to(device)
                a_tensor = a_tensor.to(device)
                d_tensor = d_tensor.to(device)
                g_tensor = g_tensor.to(device)
            scores, _ = model(x_tensor, f_tensor, a_tensor, d_tensor)
            loss_f = loss_fn(scores, g_tensor.long())
            loss_r = loss_fn(scores, torch.flip(g_tensor, (0, )).long())
            loss = torch.min(loss_f, loss_r)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.seen += 1
                if (j+1) % log_every == 0:
                    log["train"].append(dict(epoch=i+1, iter=j+1, seen=model.seen, loss=loss.item()))
                if (j+1) % print_every == 0:
                    print("epoch {} loss {:.4f}".format(i, loss.data))
                if (j+1) % val_every == 0:
                    if val_dataset is not None:
                        with torch.no_grad():
                            val_acc = val(model, val_dataset, device=device, verbose=False)
                            log["val_seen"].append(dict(seen=model.seen, acc=val_acc))
                            t2 = time.time()
                            eta = (t2 - t1) / float(j + 1) * float(len(dataset) - j - 1)
                            print("seen {}, acc {:.4f}, {:.1f}s to go for this epoch".format(model.seen, val_acc, eta))
        if val_dataset is not None:
            with torch.no_grad():
                val_acc = val(model, val_dataset, device=device, verbose=False)
                log["val"].append(dict(epoch=i, acc=val_acc))
                print("seen {}, acc {:.4f}".format(model.seen, val_acc))
        t2 = time.time()
        eta = (t2-t0) / float(i+1) * float(n_epoch-i-1)
        print("time elapsed {:.1f}s, {:.1f}s for this epoch, {:.1f}s to go".format(t2-t0, t2-t1, eta))
        print("=======================================================================================================")
    return model, log


def val(model, dataset, device=None, verbose=False, reverse_seq=False):
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
            if device is not None:
                x_tensor = x_tensor.to(device)
                f_tensor = f_tensor.to(device)
                a_tensor = a_tensor.to(device)
                d_tensor = d_tensor.to(device)
                g_tensor = g_tensor.to(device)
            _, idxs = model(x_tensor, f_tensor, a_tensor, d_tensor)
            idxs = np.array(idxs.data.cpu())
            if reverse_seq:
                _, rev_idxs = model(torch.flip(x_tensor, [0]), f_tensor, a_tensor, d_tensor)
                rev_idxs = np.array(torch.flip(rev_idxs, [0]).data.cpu())
                mask1 = torch.argmax(f_tensor[gt_idxs, :], dim=1) == torch.argmax(f_tensor[rev_idxs, :], dim=1)
                mask2 = torch.argmax(f_tensor[gt_idxs, :], dim=1) != torch.argmax(f_tensor[idxs, :], dim=1)
                mask = np.logical_and(mask1.cpu().numpy(), mask2.cpu().numpy())
                idxs[mask] = rev_idxs[mask]
            acc_f = np.sum(1 * (idxs == gt_idxs))
            acc_r = np.sum(1 * (idxs == gt_idxs[::-1]))
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
    if verbose:
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
    p.add_argument("--gpu", "-g", type=int, default=None, help="Use GPU x")
    p.add_argument("--df", type=str, default=None, help="Path to sequence database df")
    p.add_argument("--df_val", type=str, default=None, help="Path to sequence database df for validation")
    p.add_argument("--min_len", type=int, default=4, help="Minium sequence length")
    p.add_argument("--max_len", type=int, default=8, help="Maximum sequence length")
    p.add_argument("--n_lstm_hidden", type=int, default=256, help="Dimenion of LSTM hidden states")
    p.add_argument("--n_node_embed", type=int, default=128, help="Dimenion of node embeddings")
    p.add_argument("--n_seq_embed", type=int, default=32, help="Dimenion of sequence embeddings")
    p.add_argument("--graph_to_lstm", action="store_true", help="Use graph embedding for LSTM init")
    p.add_argument("--blstm", action="store_true", help="Use bidirectional LSTM")
    p.add_argument("--seed", type=int, default=2020, help="Random seed for NumPy and Pandas")
    p.add_argument("--save", "-s", type=str, default=None, help="Path and file name for saving trained model")
    p.add_argument("--model", "-m", type=str, default=None, help="Path to the pretrained model")
    p.add_argument("--log", type=str, default=None, help="Path and file name for saving training log")
    p.add_argument("--skip_training", action="store_true", help="Skip training")
    p.add_argument("--reverse_seq", action="store_true", help="Reverse sequence when validating")
    p.add_argument("--verbose", "-v", action="store_true", help="Be verbose")
    return p.parse_args()

    
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    val_dataset = GCNDataset(n=args.n_val, max_buf_size=args.max_n_seq, df_path=args.df_val,
                             seq_len_range=(args.min_len, args.max_len), seed=args.seed+1, verbose=False)
    model = GeneratorLSTM(n_graph_layers=args.n_graph_layers, device=args.gpu, n_lstm_hidden=args.n_lstm_hidden,
                          n_node_embed=args.n_node_embed, n_seq_embed=args.n_seq_embed,
                          graph_to_lstm=args.graph_to_lstm, bidirectional_lstm=args.blstm)
    model.seen = 0
    if args.model is not None:
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
        print("trained model {} loaded".format(args.model))
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
    model = model.to(device)
    if args.verbose:
        print(model)
    if args.skip_training:
        pass
    else:
        train_dataset = GCNDataset(n=args.n_train, max_buf_size=args.max_n_seq, df_path=args.df,
                                   seq_len_range=(args.min_len, args.max_len), seed=args.seed, verbose=False)
        model, json_log = train(model, train_dataset, val_dataset=val_dataset, n_epoch=args.n_epoch, lr=args.lr,
                                device=device, verbose=args.verbose)
        if args.save:
            torch.save(model.state_dict(), args.save)
        if args.log:
            json_log["params"] = args.__dict__
            with open(args.log, "w") as f:
                f.write(json.dumps(json_log))
    val_acc = val(model, val_dataset, device=device, verbose=True, reverse_seq=args.reverse_seq)
    print("test on validation set: {:.5f}".format(val_acc))


if __name__ == "__main__":
    main()
