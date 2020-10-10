import os
import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
import h5py
from torch.utils.data import Dataset
from aa_code_utils import *


def onehot(x):
    z = np.zeros((len(x), 20))
    for i, c in enumerate(x):
        z[i, c] = 1
    return z


def make_contact_map(ca_coors, mask=None, cutoff=4.0, verbose=False, sigma=None):
    # print("ca_coors", ca_coors.shape)
    if verbose:
        print("ca_coors max", np.max(ca_coors), "ca_coors min", np.min(ca_coors))
    dist_map = np.linalg.norm(ca_coors[np.newaxis, :, :] - ca_coors[:, np.newaxis, :], axis=2)
    if sigma is None:
        cont_map = dist_map < cutoff
        if mask is not None:
            cont_map[~mask, :] = False
            cont_map[:, ~mask] = False
            cont_map[~mask, ~mask] = True
    else:
        cont_map = np.exp(-0.5*np.square((dist_map-cutoff)/sigma))
        if mask is not None:
            cont_map[~mask, :] = 0.1
            cont_map[:, ~mask] = 0.1
            cont_map[~mask, ~mask] = 1.0
    if verbose:
        print("{0} contacts in the {1}x{1} contact map".format(np.sum(1.0*cont_map)-cont_map.shape[0],
                                                               cont_map.shape[0]))
    return cont_map


def make_a_matrix(idxs, self_loop=True):
    mat = (np.repeat(idxs.reshape(-1, 1), idxs.shape[0], 1) - np.repeat(idxs.reshape(1, -1), idxs.shape[0], 0))
    mat = 1.0 * (abs(mat) == 1)
    if self_loop:
        mat = mat + np.eye(len(idxs))
    return mat


def make_d_matrix(a_mat):
    mat = np.eye(a_mat.shape[0]) * np.sum(a_mat, axis=1) ** (-0.5)
    return mat


class GCNDataset(Dataset):

    def __init__(self, n, max_buf_size, df_path, seq_len_range=(128, 512), seed=0, h5=None, build_on_the_fly=False,
                 pose_feat=False, verbose=False, contact_cutoff=4.0, contact_sigma=None, coor_std=0.0,
                 dummy_ratio=0.0, dummy_mean=2.0, dummy_std=0.1):
        self.rn = np.random.RandomState(seed)
        self.n = n
        self.pose_feat = pose_feat
        self.verbose = verbose
        self.df_path = df_path
        self.df = None
        self.max_buf_size = max_buf_size
        self.seq_len_range = seq_len_range
        self.build_on_the_fly = build_on_the_fly
        self.contact_cutoff = contact_cutoff
        self.contact_sigma = contact_sigma
        self.coor_std = coor_std
        if self.df_path:
            self.df = pd.read_hdf(self.df_path, "df").query("len >= {} and len <= {}".format(*self.seq_len_range))
            if "standard" in self.df:
                self.df = self.df.query("standard")
            print("Using {} out of {} sequences".format(self.df_path, len(self.df)))
            if 0 < self.n <= len(self.df):
                self.df = self.df.sample(self.n, random_state=self.rn)
            else:
                self.n = len(self.df)
            self.use_df_data = True
            print(self.df["len"].describe())
        else:
            self.use_df_data = False
        if h5 is None:
            self.idxs = list()
            self.f = list()
            self.seq = list()
            self.gt_seq = list()
            self.gt_idxs = list()
            self.h5_filename = None
            self.h5_mode = False
        else:
            self.h5_filename = os.path.join(h5, "gcn_lstm_{}_{}_{}_{}_{}.h5".format(max_buf_size, seq_len_range[0],
                                                                                    seq_len_range[1], n, seed))
            self.h5_mode = True
        if self.build_on_the_fly:
            pass
        else:
            self.build_all()
        if self.h5_mode:
            self.h5_handle = h5py.File(self.h5_filename, "r")
        self.dummy_ratio = dummy_ratio
        self.dummy_mean = dummy_mean
        self.dummy_std = dummy_std

    def make_dummy_ca(self, ca_coors):
        n = ca_coors.shape[0]
        n_dummy = np.random.randint(n, int(self.dummy_ratio-1) * n + 1)
        ref_idxs = np.random.choice(n, size=n_dummy, replace=True)
        dummy_coors = ca_coors[ref_idxs, :]
        dummy_offsets = np.random.randn(*dummy_coors.shape)
        dummy_mags = self.dummy_mean + self.dummy_std * np.random.randn(n_dummy, 1)
        dummy_offsets = dummy_mags / np.linalg.norm(dummy_offsets, axis=1, keepdims=True) * dummy_offsets
        dummy_coors += dummy_offsets
        ca_coors = np.concatenate((ca_coors, dummy_coors), axis=0)
        return ca_coors

    def build_one(self, idx):
        my_a_mat = None
        my_ca = None
        my_mask = None
        make_dummy = True
        if self.use_df_data:
            my_seq_len = self.df["len"].iloc[idx]
            my_seq = np.array([a2id(c) for c in self.df["seq"].iloc[idx]])
            if "CA_coors" in self.df:
                my_ca = self.df["CA_coors"].iloc[idx] * 0.01  # ProteinNet's coordinates are in pm
                my_mask = self.df["mask"].iloc[idx]
                if self.dummy_ratio > 1:
                    my_ca = self.make_dummy_ca(my_ca)
                    my_mask = np.concatenate((my_mask, np.ones((my_ca.shape[0]-my_mask.shape[0], )) > 0), axis=0)
                if self.coor_std > 0:
                    my_ca += self.coor_std * np.random.randn(*my_ca.shape)
                my_a_mat = 1.0 * make_contact_map(my_ca, my_mask, cutoff=self.contact_cutoff)
                if self.verbose:
                    print("contact map created", my_a_mat.shape)
                make_dummy = False
                my_mask = my_mask.astype(np.int)
                # print("my_mask", my_mask)
                if self.pose_feat:
                    my_c = self.df["C_coors"].iloc[idx] * 0.01
                    my_n = self.df["N_coors"].iloc[idx] * 0.01
                    c_ca = my_c - my_ca
                    n_ca = my_n - my_ca
                    c_ca = c_ca / np.linalg.norm(c_ca, axis=1, keepdims=True)
                    c_ca[np.isnan(c_ca)] = 0
                    n_ca = n_ca / np.linalg.norm(n_ca, axis=1, keepdims=True)
                    n_ca[np.isnan(n_ca)] = 0
                    # print("c_ca", np.max(c_ca), "n_ca", np.max(n_ca))
        else:
            my_seq_len = self.rn.randint(self.seq_len_range[0], self.seq_len_range[1])
            my_seq = self.rn.randint(0, 20, my_seq_len)
        my_seq_idxs = np.arange(my_seq_len)
        if self.verbose:
            print("my_seq_idxs", my_seq_idxs.shape)
        if make_dummy:
            my_dummy_idxs = self.rn.permutation(np.arange(self.seq_len_range[1], 2 * self.max_buf_size))[
                            0:(self.max_buf_size - my_seq_len)]
            if self.verbose:
                print("my_dummy_idxs", my_dummy_idxs.shape)
            my_rand_idxs = self.rn.permutation(self.max_buf_size)
            if self.verbose:
                print("my_rand_idxs", my_rand_idxs.shape)
            my_idxs = np.concatenate((my_seq_idxs, my_dummy_idxs))[my_rand_idxs]
            if self.verbose:
                print("my_idxs", my_idxs.shape)
            my_seq = np.concatenate((my_seq, self.rn.randint(0, 20, self.max_buf_size - my_seq_len)))
            my_seq = my_seq[my_rand_idxs]
        else:
            my_idxs = np.arange(my_ca.shape[0])
        my_feats = 0.01 * abs(self.rn.randn(my_ca.shape[0], 20))
        my_gt_idxs = np.argsort(my_idxs)[0:my_seq_len]
        my_gt_seq = my_seq[my_gt_idxs]
        for j, s in enumerate(my_seq):
            my_feats[j, s] = 1 - np.sum(my_feats[j, :]) + my_feats[j, s]
        if self.pose_feat:
            my_feats = np.concatenate((my_feats, c_ca, n_ca), axis=1)
        return my_idxs, my_feats, my_seq, my_gt_seq, my_gt_idxs, my_a_mat, my_ca, my_mask

    def build_all(self):
        if self.h5_mode:
            handle = h5py.File(self.h5_filename, "w")
        for i in range(self.n):
            my_idxs, my_feats, my_seq, my_gt_seq, my_gt_idxs, _, _, _ = self.build_one(i)
            if self.h5_mode:
                handle.create_dataset("idxs_{}".format(i), my_idxs.shape, data=my_idxs)
                handle.create_dataset("f_{}".format(i), my_feats.shape, data=my_feats)
                handle.create_dataset("seq_{}".format(i), my_seq.shape, data=my_seq)
                handle.create_dataset("gt_seq_{}".format(i), my_gt_seq.shape, data=my_gt_seq)
                handle.create_dataset("gt_idxs_{}".format(i), my_gt_idxs.shape, data=my_gt_idxs)
            else:
                self.idxs.append(my_idxs)
                self.f.append(my_feats)
                self.seq.append(my_seq)
                self.gt_seq.append(my_gt_seq)
                self.gt_idxs.append(my_gt_idxs)
            if (i+1) % 10000 == 0:
                print("processed {} sequences".format(i+1))
        if self.h5_mode:
            handle.close()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        a_mat = None
        ca_coor = None
        ca_mask = None
        if self.build_on_the_fly:
            my_idxs, f, my_seq, my_gt_seq, gt_idxs, a_mat, ca_coor, ca_mask = self.build_one(idx)
        elif self.h5_mode:
            my_gt_seq = self.h5_handle["gt_seq_{}".format(idx)][()]
            my_idxs = self.h5_handle["idxs_{}".format(idx)][()]
            f = self.h5_handle["f_{}".format(idx)][()]
            gt_idxs = self.h5_handle["gt_idxs_{}".format(idx)][()]
        else:
            my_gt_seq = self.gt_seq[idx]
            my_idxs = self.idxs[idx]
            f = self.f[idx]
            gt_idxs = self.gt_idxs[idx]
        x = onehot(my_gt_seq)
        if a_mat is None:
            a_mat = make_a_matrix(my_idxs)
            ca_coor = np.zeros((a_mat.shape[0], 3))
            ca_mask = np.zeros((a_mat.shape[0], 1))
        d_mat = make_d_matrix(a_mat)
        return x, f, a_mat, d_mat, gt_idxs, ca_coor, ca_mask
