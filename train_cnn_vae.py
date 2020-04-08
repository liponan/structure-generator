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
             "#"]
    return table[aid]


class SeqDataset(Dataset):

    def __init__(self, n, df_path, max_seq_len=1024, seq_len_range=(128, 512), verbose=False):
        self.n = n
        self.max_seq_len = max_seq_len
        self.verbose = verbose
        self.df_path = df_path
        self.idxs = list()
        self.seq = list()
        self.gt_seq = list()
        self.gt_idxs = list()
        self.build(seq_len_range)

    def build(self, seq_len_range):
        if self.df_path:
            df = pd.read_hdf(self.df_path, "df").query("len >= {} and len <= {} and standard".format(seq_len_range[0], seq_len_range[1]))
            print("using {} ({} sequences)".format(self.df_path, len(df)))
            if 0 < self.n < len(df):
                df = df.sample(self.n)
            else:
                self.n = len(df)
        for i in range(self.n):
            if self.df_path:
                my_seq_len = df["len"].iloc[i]
            else:
                my_seq_len = np.random.randint(seq_len_range[0], seq_len_range[1])
            if self.df_path:
                my_seq = np.array([a2id(c) for c in df["seq"].iloc[i]])
            else:
                my_seq = np.random.randint(0, 21, my_seq_len)
            my_seq = np.concatenate((my_seq, np.array([20])))
            self.gt_seq.append(my_seq)

    def onehot(self, x):
        z = np.zeros((len(x), 21))
        for i, c in enumerate(x):
            z[i, int(c)] = 1
        return z

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        my_gt_seq = 20 * np.ones((self.max_seq_len, ))
        my_gt_seq[0:len(self.gt_seq[idx])] = self.gt_seq[idx]
        x = self.onehot(my_gt_seq)
        return x, my_gt_seq


class FCVAE(nn.Module):

    def __init__(self, n_input, n_embed, n_h1, n_h2, n_h3, max_seq_len):
        super(CNNVAE, self).__init__()
        self.seq2embed = nn.Linear(n_input, n_embed)
        self.w1 = nn.Linear(n_embed*max_seq_len, n_h1)
        self.w2 = nn.Linear(n_h1, n_h2)
        self.w3 = nn.Linear(n_h2, n_h3)
        self.w4 = nn.Linear(n_h3, n_h2)
        self.w5 = nn.Linear(n_h2, n_h1)
        self.w6 = nn.Linear(n_h1, n_embed*max_seq_len)
        self.embed2seq = nn.Linear(n_embed, n_input)
        self.n_embed = n_embed
        self.max_seq_len = max_seq_len
        self.relu = nn.ReLU()

    def forward(self, seq):
        x = self.seq2embed(seq).flatten()
        x = self.w1(x)
        x = self.w2(x)
        x = self.w3(x)
        if self.train:
            z = self.w4(x)
            z = self.w5(z)
            z = self.w6(z).view(-1, self.max_seq_len, self.n_embed)
            z = self.embed2seq(z)
            return z
        else:
            return x


class CNNVAE(nn.Module):

    def __init__(self, n_input, n_embed, n_h1, n_h2, n_h3, max_seq_len):
        super(CNNVAE, self).__init__()
        self.seq2embed = nn.Linear(n_input, n_embed)
        self.w1 = nn.Linear(n_embed*max_seq_len, n_h1)
        self.w2 = nn.Linear(n_h1, n_h2)
        self.w3 = nn.Linear(n_h2, n_h3)
        self.w4 = nn.Linear(n_h3, n_h2)
        self.w5 = nn.Linear(n_h2, n_h1)
        self.w6 = nn.Linear(n_h1, n_embed*max_seq_len)
        self.embed2seq = nn.Linear(n_embed, n_input)
        self.n_embed = n_embed
        self.max_seq_len = max_seq_len
        self.relu = nn.ReLU()

    def forward(self, seq):
        x = self.seq2embed(seq).flatten()
        x = self.w1(x)
        x = self.w2(x)
        x = self.w3(x)
        if self.train:
            z = self.w4(x)
            z = self.w5(z)
            z = self.w6(z).view(-1, self.max_seq_len, self.n_embed)
            z = self.embed2seq(z)
            return z
        else:
            return x


def train(model, dataset, n_epoch=1, lr=0.1, gpu=False):
    print("====================== train ======================")
    loss_fn = nn.CrossEntropyLoss()
    for i in range(n_epoch):
        if i % 50 == 0 and i > 0:
            lr /= 10
            print("lr is now {}".format(lr))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for j in range(len(dataset)):
            model.zero_grad()
            x, y = dataset[j]
            x_tensor = torch.from_numpy(x).float().view(1, -1, 21)
            y_tensor = torch.from_numpy(y).long().view(1, -1)
            if gpu:
                x_tensor = x_tensor.cuda()
            scores = model(x_tensor).transpose(1, 2)
            loss = loss_fn(scores, y_tensor)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pd_idxs = torch.argmax(scores.cpu().data, dim=1)
            print("GT", "".join([id2a(int(i)) for i in y]))
            print("Pr", "".join([id2a(int(i)) for i in pd_idxs.numpy().reshape(-1)]))
        print("epoch {} loss {:.4f}".format(i, loss.data))
    return model


def val(model, dataset, gpu=False, verbose=False):
    print("======================  val  ======================")
    acc_all = 0
    count = 0
    model.eval()
    for j in range(len(dataset)):
        x, y = dataset[j]
        x_tensor = torch.from_numpy(x).float().view(1, -1, 21)
        y_tensor = torch.from_numpy(y).long().view(-1)
        if gpu:
            x_tensor = x_tensor.cuda()
        scores = model(x_tensor).transpose(1, 2)
        pd_idxs = torch.argmax(scores.cpu().data, dim=1).view(-1).numpy()
        if verbose:
            print("GT", "".join([id2a(int(i)) for i in y]))
            print("Pr", "".join([id2a(int(i)) for i in pd_idxs]))
        mask = y != 20
        n_gt = np.sum(1*mask)
        acc = np.sum(pd_idxs[mask]==y[mask]) / float(n_gt)
        acc_all += acc
        count += 1
    acc_all = acc_all / float(count)
    print("over all accuracy", acc_all)
            

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_train", "-n", type=int, default=10, help="Number of training samples")
    p.add_argument("--n_val", type=int, default=10, help="Number of val samples")
    p.add_argument("--n_epoch", "-e", type=int, default=1, help="Number of training epoch")
    p.add_argument("--min_len", type=int, default=16, help="Minium sequence length")
    p.add_argument("--max_len", type=int, default=31, help="Maximum sequence length")
    p.add_argument("--buffer_size", type=int, default=32, help="Input buffer size")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--df", type=str, default=None, help="Path to sequence database df")
    p.add_argument("--df_val", type=str, default=None, help="Path to sequence database df for validation")
    p.add_argument("--gpu", "-g", action="store_true", help="Use GPU")
    return p.parse_args()

    
def main():
    args = parse_args()
    train_dataset = SeqDataset(n=args.n_train, df_path=args.df, max_seq_len=args.buffer_size,
                               seq_len_range=(args.min_len, args.max_len), verbose=False)
    val_dataset = SeqDataset(n=args.n_val, df_path=args.df_val, max_seq_len=args.buffer_size,
                             seq_len_range=(args.min_len, args.max_len), verbose=False)
    model = CNNVAE(n_input=21, n_embed=32, n_h1=1024, n_h2=512, n_h3=256, max_seq_len=args.buffer_size)
    if args.gpu:
        model = model.cuda()
    model = train(model, train_dataset, n_epoch=args.n_epoch, lr=args.lr, gpu=args.gpu)
    val(model, train_dataset, gpu=args.gpu)
    val(model, val_dataset, gpu=args.gpu, verbose=True)


if __name__ == "__main__":
    main()
