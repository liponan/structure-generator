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
from lstm_utils import *
from aa_code_utils import *


class LSTMDataset(Dataset):
    
    def __init__(self, n, max_seq_size, n_features=20, n_dict=20, cuda=False):
        self.n = n
        self.max_seq_size = max_seq_size
        self.n_features = n_features
        self.n_dict = n_dict
        self.cuda = cuda
        self.seqs = np.random.randint(0, 19, (n, self.max_seq_size))

    def onehot(self, x):
        z = torch.zeros(len(x), self.n_dict)
        for i, c in enumerate(x):
            z[i, c] = 1
        return z
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        seq = self.seqs[idx, :]
        x = self.onehot(seq)
        rand_idx = np.random.permutation(len(seq))
        y = torch.from_numpy(np.argsort(np.arange(len(seq))[rand_idx]))
        A = (np.repeat(rand_idx.reshape(-1, 1), rand_idx.shape[0], 1) - np.repeat(rand_idx.reshape(1, -1), rand_idx.shape[0], 0))
        A = 1.0 * (abs(A) == 1)
        B = 1*np.random.rand(rand_idx.shape[0], self.n_features)
        for i in range(B.shape[0]):
            j = seq[rand_idx[i]]
            B[i,j] = B[i,j] + 1
        B = B / np.sum(B, axis=1, keepdims=True)
        C = torch.cat((torch.from_numpy(A), torch.from_numpy(B)), dim=1).float()
        if self.cuda:
            C = C.cuda()
            x = x.cuda()
            y = y.cuda()
        return C, x, y
    
    
class LSTMSeq(nn.Module):

    def __init__(self, feature_dim, embedding_dim, hidden_dim, tagset_size):
        super(LSTMSeq, self).__init__()
        self.fc = nn.Linear(feature_dim, 2*hidden_dim)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sequence, features):
        fc_out = self.fc(features.view(-1))
        h0 = fc_out[0:self.hidden_dim]
        c0 = fc_out[self.hidden_dim:]
        lstm_out, _ = self.lstm(sequence.view(-1, 1, self.embedding_dim), (h0.view(1,1,-1), c0.view(1,1,-1)))
        tag_out = self.hidden2tag(lstm_out.view(-1, self.hidden_dim))
        return tag_out

    
def train(model, dataset, n_epoch=1, lr=0.1):
    print("====================== train ======================")
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for i in range(n_epoch):
        for j in range(len(dataset)):
            features, x, y = dataset[j]
            model.zero_grad()
            scores = model(x, features)
            loss = loss_fn(scores, y.long())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            print("epoch {} loss {:.4f}".format(i, loss.data))
     
    return model


def val(model, dataset):
    print("======================  val  ======================")
    model.eval()
    with torch.no_grad():
        for j in range(len(dataset)):
            out = model(dataset[j][1], dataset[j][0])
            idx = torch.argmax(out.cpu().data, dim=1)
            print(j, np.sum(1*(dataset[j][2].cpu().data.numpy() == idx.numpy())))
            

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_train", "-n", type=int, default=10, help="Number of training samples")
    p.add_argument("--n_epoch", "-e", type=int, default=100, help="Number of training epoch")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    p.add_argument("--gpu", "-g", action="store_true", help="Use GPU")
    return p.parse_args()

    
def main():
    args = parse_args()
    dataset = LSTMDataset(args.n_train, 128, cuda=args.gpu)
    model = LSTMSeq(148*128, 20, 1024, 128)
    if args.gpu:
        model = model.cuda()
    model = train(model, dataset, n_epoch=args.n_epoch, lr=args.lr)
    val(model, dataset)


if __name__ == "__main__":
    main()
