import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import h5py
from torch.utils.data import Dataset
from aa_code_utils import *
from dataset import onehot, make_d_matrix


class Rotamer2AA(object):

    def __init__(self):
        self.mat = np.load("rotamer2aa_ukn.npy")

    def __call__(self, x):
        x = np.exp(x)
        x = x @ self.mat
        x = np.max(x.reshape(-1, 21, 34), axis=2)[:, 0:20]
        x = x / np.sum(x, axis=1, keepdims=True)
        x[np.isnan(x)] = 0
        return x


def load_seq(seq_path):
    with open(seq_path, "r") as f:
        seq = list(f.read().strip())
    return aa1toidx(seq)


class DeepMapDataset(Dataset):

    def __init__(self, data_path, ca_ca_cutoff=4.5):
        self.n = -1
        self.files = None
        self.ca_ca_cutoff = ca_ca_cutoff
        self._build(data_path)


    def _build(self, data_path):
        self.files = sorted(glob.glob(os.path.join(data_path, "*_*.csv")))
        self.n = len(self.files)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        df = pd.read_csv(self.files[idx], index_col=0)
        x = onehot(load_seq(self.files[idx].replace(".csv", ".txt")))
        ca_coors = df[["x", "y", "z"]].values
        f = df.iloc[:, 4:].values
        a_mat = np.linalg.norm(
            ca_coors[np.newaxis, :, :] - ca_coors[:, np.newaxis, :],
            axis=2
        )
        a_mat = 1.0 * (a_mat < self.ca_ca_cutoff)
        d_mat = make_d_matrix(a_mat)
        gt_idxs = np.arange(x.shape[0])
        ca_mask = np.ones((a_mat.shape[0],))
        print("[{}] seq {} mat {} {}".format(
            idx, x.shape[0], a_mat.shape[0], self.files[idx]
        ))
        return x, f, a_mat, d_mat, gt_idxs, ca_coors, ca_mask
