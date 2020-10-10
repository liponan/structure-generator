import os
import glob
import time
import json
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from aa_code_utils import *
from dataset import GCNDataset
from networks import GeneratorLSTM, FocalLoss
from dataset import onehot, make_d_matrix
import argparse
import sys
from dataset import onehot
sys.path.append("../m2m")
from train_alphanet import AlphaNet, AAAccuracy, metrics, make_cat_map, make_field_map, make_vector_map,\
    data_noramlization, transform_data, rotate_data


class AlphaSeqDataset(Dataset):

    def __init__(self, dataset_df_path, normalize=False, transform=False, downsample=1, n=-1, max_size=-1,
                 query=None, seed=2020):
        self.rg = np.random.RandomState(seed)
        self.files = None
        self.df = pd.read_hdf(dataset_df_path, "df")
        if query is not None:
            self.df = self.df.query(query)
        if max_size > 0:
            self.df = self.df.query("i <= {0} and j <= {0} and k <= {0}".format(max_size))
        if 0 < n <= len(self.df):
            self.df = self.df.sample(n)
        self.df = self.df.reset_index(drop=True)
        self.n = len(self.df)
        self.normalize = normalize
        self.transform = transform
        self.downsample = downsample

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # print("getitem", idx)
        attr_path, mask_path = self.df.iloc[idx][["attr_path", "mask_path"]]
        with h5py.File(mask_path, "r") as f:
            data = f["data"][()]
        if self.normalize:
            data = data_noramlization(data)
        df_attr = pd.read_hdf(attr_path, "df")
        df_attr = df_attr.query("chain == '{}'".format(df_attr.iloc[0]["chain"])).sort_values(by="no")
        atom_coors = df_attr[["O", "C", "CA", "N"]].dropna(how="any", axis=0).values
        o_coors = np.concatenate(atom_coors[:, 0]).reshape(-1, 3)
        c_coors = np.concatenate(atom_coors[:, 1]).reshape(-1, 3)
        ca_coors = np.concatenate(atom_coors[:, 2]).reshape(-1, 3)
        n_coors = np.concatenate(atom_coors[:, 3]).reshape(-1, 3)
        o_field = make_field_map(o_coors, ca_coors, data.shape, downsample=self.downsample)
        c_field = make_field_map(c_coors, ca_coors, data.shape, downsample=self.downsample)
        ca_field = make_vector_map(ca_coors, data.shape, downsample=self.downsample)
        n_field = make_field_map(n_coors, ca_coors, data.shape, downsample=self.downsample)
        r_tokens = rot_to_idx(list(df_attr["rot_label"].values))
        r_labels = make_cat_map(r_tokens, ca_coors, data.shape, downsample=self.downsample)
        a_tokens = aa3toidx(list(df_attr["aa_label"].values))
        a_labels = make_cat_map(a_tokens, ca_coors, data.shape, downsample=self.downsample)
        i_tokens = list(df_attr["no"].values)
        i_labels = make_cat_map(i_tokens, ca_coors, data.shape, downsample=self.downsample)
        data = data[np.newaxis, :, :, :]
        if self.transform:
            transform_seqs = self.rg.randint(0, 4, (3,))
            data = transform_data(data, transform_seqs).copy()
            o_field = transform_data(o_field, transform_seqs).copy()
            c_field = transform_data(c_field, transform_seqs).copy()
            ca_field = transform_data(ca_field, transform_seqs).copy()
            n_field = transform_data(n_field, transform_seqs).copy()
            r_labels = transform_data(r_labels, transform_seqs).copy()
            a_labels = transform_data(a_labels, transform_seqs).copy()
            i_labels = transform_data(i_labels, transform_seqs).copy()
        seq = onehot(a_tokens)
        n_gt = int(np.sum(ca_field[0, :, :, :]))
        if n_gt != seq.shape[0]:
            print("n_gt", n_gt, "seq.shape[0]", seq.shape[0])
            seq = seq[:n_gt, :]
        return data, (o_field, c_field, ca_field, n_field, r_labels, a_labels, i_labels), seq


def extract(scores, cutoff=0.1, downsample=8, dx=0.25):
    ca_scores = nn.Sigmoid()(scores[0:4, :, :, :].data)
    mask = ca_scores[0, :, :, :] > cutoff
    # print("mask", mask.size())
    ca_conf = ca_scores[0, :, :, :][mask].view(-1, 1)
    # print("conf", conf.size())
    uvw = torch.nonzero(mask)
    # print("uwv", uvw.size())
    u = ca_scores[1, :, :, :][mask] + uvw[:, 0].float()
    v = ca_scores[2, :, :, :][mask] + uvw[:, 1].float()
    w = ca_scores[3, :, :, :][mask] + uvw[:, 2].float()
    ca_coors = downsample * dx * torch.cat((u.view(-1, 1), v.view(-1, 1), w.view(-1, 1)), dim=1)
    # print("ca_coors", ca_coors.size())
    cat = scores[13:, :, :, :][mask[None, :, :, :].expand(163, mask.size(0), mask.size(1), mask.size(2))]
    cat = torch.transpose(cat.view(163, -1), 0, 1)
    # print("cat", cat.size())
    return ca_conf, ca_coors, cat


def extract_gt_coors(ca_labels, idx_labels, downsample=8, dx=0.25):
    mask = ca_labels[0, :, :, :] > 0
    uvw = torch.nonzero(mask)
    # print("uwv", uvw.size())
    u = ca_labels[1, :, :, :][mask].float() + uvw[:, 0].float()
    v = ca_labels[2, :, :, :][mask].float() + uvw[:, 1].float()
    w = ca_labels[3, :, :, :][mask].float() + uvw[:, 2].float()
    gt_coors = downsample * dx * torch.cat((u.view(-1, 1), v.view(-1, 1), w.view(-1, 1)), dim=1)
    _, idxs = torch.sort(idx_labels[0, :, :, :][mask])
    gt_coors = gt_coors[idxs, :]
    return gt_coors


def extract_gt_idxs(ca_labels, idx_labels, downsample=8, dx=0.25):
    mask = ca_labels[0, :, :, :] > 0
    _, gt_idxs = torch.sort(idx_labels[0, :, :, :][mask])
    return gt_idxs


def train(model1, model2, ds, params, device1, device2, conf_cutoff=0.1, ca_cutoff=4.0, dist_loss_weight=0.0, verbose=False):
    n_iters = int(len(ds) / params["batch_size"])
    dataloader = DataLoader(ds, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"],
                            drop_last=False)
    matrix_path = "rotamer2aa_ukn.npy"
    aaacc = AAAccuracy(matrix_path=matrix_path, device=device1)
    rot2aa = torch.from_numpy(np.load(matrix_path, allow_pickle=True)).to(device2).float()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr=params["lr"])
    seen = 0
    t1 = time.time()
    for j, (x, y, z) in enumerate(dataloader):
        x = x.to(device1)
        o_field = y[0].to(device1)
        c_field = y[1].to(device1)
        ca_field = y[2].to(device1)
        n_field = y[3].to(device1)
        rot_labels = y[4].to(device1)
        aa_labels = y[5].to(device1)
        idx_labels = y[6].to(device1)
        z = z[0, :, :].to(device2)
        with torch.no_grad():
            # model1.eval()
            model1.train()
            output1 = model1(x)
            n_gt, n_p, pre, rec, o_err, c_err, ca_err, n_err, rot_cat_acc, aa_cat_acc \
                = metrics(output1, o_field.float(), c_field.float(), ca_field.float(), n_field.float(),
                          rot_labels.long(), aa_labels, aaacc=aaacc)
            if verbose:
                print("*  GT {:3d}  detected {:3d}  precision {:.3f}  recall {:.3f}  rot {:.3f}  aa {:.3f}"
                      .format(n_gt, n_p, pre, rec, rot_cat_acc, aa_cat_acc))
                print("*  O {:.3f}  C {:.3f}  CA {:.3f}  N {:.3f}".format(o_err, c_err, ca_err, n_err))
            ca_conf, ca_coors, cat_scores = extract(output1[0, :, :, :, :], cutoff=conf_cutoff)
            ca_conf.to(device2)
            ca_coors.to(device2)
            cat_scores.to(device2)
            aa_scores = torch.max(torch.mm(torch.exp(cat_scores), rot2aa).view(-1, 21, 34), dim=2)[0][:, :-1]
            # aa_scores = torch.exp(aa_scores)
            aa_scores = aa_scores / torch.sum(aa_scores, dim=1)[:, None]
            gt_coors = extract_gt_coors(ca_field[0, :, :, :, :], idx_labels[0, :, :, :, :])
            dist_mask = torch.min(torch.norm(gt_coors[None, :, :] - ca_coors[:, None, :], dim=2), dim=1)[0] < 2.00
            ca_coors = ca_coors[dist_mask, :]
            aa_scores = aa_scores[dist_mask, :]
            ca_dists = torch.norm(ca_coors[:, None, :] - ca_coors[None, :, :], dim=2)
            adj_mat = (ca_dists < ca_cutoff).float()
            d_mat = torch.eye(adj_mat.size(0)).to(device2).float() * torch.pow(torch.sum(adj_mat, dim=1), -0.5)
            # gt_idxs = extract_gt_idxs(ca_field[0, :, :, :, :], idx_labels[0, :, :, :])
            # gt_idxs = torch.argmin(torch.norm(gt_coors[:, None, :] - ca_coors[None, :, :], dim=2), dim=1)
            gt_idxs = torch.argmin(torch.norm(torch.floor(gt_coors[:, None, :]/2.0) -
                                              torch.floor(ca_coors[None, :, :]/2.0), dim=2), dim=1)
        model2.train()
        model2.zero_grad()
        # print("z", z.size())
        # print("aa_scores", aa_scores.size())
        # print("adj_mat", adj_mat.size())
        # print("d_mat", d_mat.size())
        ca_mask = torch.ones(ca_coors.size(0), ).to(device2)
        output2, idxs = model2(z.float(), aa_scores, adj_mat, d_mat, coor_tensor=ca_coors, mask_tensor=ca_mask)
        model2.seen += 1
        loss1 = loss_fn(output2[:gt_idxs.size(0), :], gt_idxs.long())
        dists = torch.norm(ca_coors[idxs[0:-1]] - ca_coors[idxs[1:]], dim=1)
        # loss2 = nn.MSELoss(torch.clamp(dists, min=4.5) - 4.5)
        loss2 = torch.sum(torch.abs(torch.clamp(dists, min=4.5) - 4.5))
        loss = loss1 + dist_loss_weight * loss2
        if model2.seen % params["back_every"] == 0:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            gt_seq = torch.argmax(z, dim=1)
            pr_seq = torch.argmax(aa_scores[idxs, :], dim=1)
            seq_acc = float(torch.sum(gt_seq == pr_seq).data.cpu())
            seq_acc = seq_acc / float(len(idxs))
            gt_coors = gt_coors.data.cpu().numpy()
            ca_coors = ca_coors.data.cpu().numpy()
            np_idxs = idxs.data.cpu().numpy()
            rmsd = float(np.sqrt(np.sum(np.square(gt_coors - ca_coors[np_idxs, :])) / gt_coors.shape[0]))
            min_dist = float(np.min(np.linalg.norm(gt_coors - ca_coors[np_idxs, :], axis=1)))
            median_dist = float(np.median(np.linalg.norm(gt_coors - ca_coors[np_idxs, :], axis=1)))
            max_dist = float(np.max(np.linalg.norm(gt_coors - ca_coors[np_idxs, :], axis=1)))
            precision = float(torch.sum(idxs == gt_idxs)) / float(len(gt_idxs))
            seen += 1
            if verbose:
                print("GT", "".join([id2a(c) for c in list(gt_seq.data.cpu())]))
                print("PR", "".join([id2a(c) for c in list(pr_seq.data.cpu())]))
                print("{:3d}: acc {:.3f}  precision {:.3f}  rmsd {:.3f}  min {:.3f}  median {:.3f}  max {:.3f}"
                      .format(seen, seq_acc, precision, rmsd, min_dist, median_dist, max_dist))
            if model2.seen % params["save_every"] == 0:
                output_filename = "backup/alphanet/latest.pt"
                torch.save(model2.state_dict(), output_filename)
    return model2


def val(model1, model2, ds, params, device1, device2, conf_cutoff=0.1, ca_cutoff=4.0, use_tp=False, nms=0,
        verbose=False):
    n_iters = int(len(ds) / params["batch_size"])
    dataloader = DataLoader(ds, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"],
                            drop_last=False)
    matrix_path = "rotamer2aa_ukn.npy"
    aaacc = AAAccuracy(matrix_path=matrix_path, device=device1)
    rot2aa = torch.from_numpy(np.load(matrix_path, allow_pickle=True)).to(device2).float()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr=params["lr"])
    seen = 0
    acc_seq_accuracy = 0
    acc_gen_accuracy = 0
    acc_data = list()
    t1 = time.time()
    with torch.no_grad():
        for j, (x, y, z) in enumerate(dataloader):
            x = x.to(device1)
            o_field = y[0].to(device1)
            c_field = y[1].to(device1)
            ca_field = y[2].to(device1)
            n_field = y[3].to(device1)
            rot_labels = y[4].to(device1)
            aa_labels = y[5].to(device1)
            idx_labels = y[6].to(device1)
            z = z[0, :, :].to(device2)
            # model1.eval()
            model1.train()
            output1 = model1(x)
            n_gt, n_p, pre, rec, o_err, c_err, ca_err, n_err, rot_cat_acc, aa_cat_acc \
                = metrics(output1, o_field.float(), c_field.float(), ca_field.float(), n_field.float(),
                          rot_labels.long(), aa_labels, aaacc=aaacc)
            if verbose:
                print("*  GT {:3d}  detected {:3d}  precision {:.3f}  recall {:.3f}  rot {:.3f}  aa {:.3f}"
                      .format(n_gt, n_p, pre, rec, rot_cat_acc, aa_cat_acc))
                print("*  O {:.3f}  C {:.3f}  CA {:.3f}  N {:.3f}".format(o_err, c_err, ca_err, n_err))
            ca_conf, ca_coors, cat_scores = extract(output1[0, :, :, :, :], cutoff=conf_cutoff)
            if ca_coors.size(0) < 2:
                print("two few candidates... skipped")
                continue
            ca_conf.to(device2)
            ca_coors.to(device2)
            cat_scores.to(device2)
            aa_scores = torch.max(torch.mm(torch.exp(cat_scores), rot2aa).view(-1, 21, 34), dim=2)[0][:, :-1]
            # aa_scores = torch.exp(aa_scores)
            aa_scores = aa_scores / torch.sum(aa_scores, dim=1)[:, None]
            gt_coors = extract_gt_coors(ca_field[0, :, :, :, :], idx_labels[0, :, :, :, :])
            if use_tp:
                _, gt_corr_idxs = torch.min(torch.norm(torch.floor(gt_coors[None, :, :]/2.0) -
                                                       torch.floor(ca_coors[:, None, :]/2.0), dim=2), dim=0)
                # scenraio 1: only use true positive nodes
                # ca_coors = gt_coors
                # aa_scores = aa_scores[gt_corr_idxs, :]
                # scenraio 2: replace TP's coors with GT coors
                # ca_coors[gt_corr_idxs, :] = gt_coors
                # scenraio 3: only use true positive nodes, but keep predicted coors
                ca_coors = ca_coors[gt_corr_idxs, :]
                ca_conf = ca_conf[gt_corr_idxs, :]
                aa_scores = aa_scores[gt_corr_idxs, :]
            else:
                dist_mask = torch.min(torch.norm(gt_coors[None, :, :] - ca_coors[:, None, :], dim=2), dim=1)[0] < 2.00
                ca_coors = ca_coors[dist_mask, :]
                ca_conf = ca_conf[dist_mask, :]
                if ca_coors.size(0) < 2:
                    print("two few candidates after dist_mask... skipped")
                    continue
                aa_scores = aa_scores[dist_mask, :]
            if nms > 0:
                n0 = int(ca_coors.size(0))
                sorted_idxs = torch.argsort(ca_conf, descending=True, dim=0)
                # print(sorted_idxs)
                new_idxs = list([int(sorted_idxs[0])])
                for k in range(1, sorted_idxs.size(0)):
                    min_dist = torch.min(torch.norm(ca_coors[sorted_idxs[k], None, :] - ca_coors[None, new_idxs, :],
                                                    dim=2))
                    # print(k, min_dist)
                    if min_dist > nms:
                        new_idxs.append(int(sorted_idxs[k]))
                    else:
                        continue
                n1 = len(new_idxs)
                print("NMS: {} => {}".format(n0, n1))
                ca_coors = ca_coors[new_idxs, :]
                aa_scores = aa_scores[new_idxs, :]
            ca_dists = torch.norm(ca_coors[:, None, :] - ca_coors[None, :, :], dim=2)
            adj_mat = (ca_dists < ca_cutoff).float()
            d_mat = torch.eye(adj_mat.size(0)).to(device2).float() * torch.pow(torch.sum(adj_mat, dim=1), -0.5)
            # gt_idxs = extract_gt_idxs(ca_field[0, :, :, :, :], idx_labels[0, :, :, :])
            # gt_idxs = torch.argmin(torch.norm(gt_coors[:, None, :] - ca_coors[None, :, :], dim=2), dim=1)
            gt_idxs = torch.argmin(torch.norm(torch.floor(gt_coors[:, None, :] / 2.0) -
                                              torch.floor(ca_coors[None, :, :] / 2.0), dim=2), dim=1)
            model2.eval()
            # print("z", z.size())
            # print("aa_scores", aa_scores.size())
            # print("adj_mat", adj_mat.size())
            # print("d_mat", d_mat.size())
            ca_mask = torch.ones(ca_coors.size(0), ).to(device2) > 0
            # print("ca_coor", ca_coors.size())
            # print("ca_mask", ca_mask.size())
            output2, idxs = model2(z.float(), aa_scores, adj_mat, d_mat, coor_tensor=ca_coors, mask_tensor=ca_mask)
            model2.seen += 1
            gt_seq = torch.argmax(z, dim=1)
            pr_seq = torch.argmax(aa_scores[idxs, :], dim=1)
            seq_acc = float(torch.sum(gt_seq == pr_seq).data.cpu())
            seq_acc = seq_acc / float(len(idxs))
            gt_coors = gt_coors.data.cpu().numpy()
            ca_coors = ca_coors.data.cpu().numpy()
            np_idxs = idxs.data.cpu().numpy()
            rmsd = float(np.sqrt(np.sum(np.square(gt_coors - ca_coors[np_idxs, :])) / gt_coors.shape[0]))
            min_dist = float(np.min(np.linalg.norm(gt_coors - ca_coors[np_idxs, :], axis=1)))
            median_dist = float(np.median(np.linalg.norm(gt_coors - ca_coors[np_idxs, :], axis=1)))
            max_dist = float(np.max(np.linalg.norm(gt_coors - ca_coors[np_idxs, :], axis=1)))
            gen_acc = float(torch.sum(idxs == gt_idxs)) / float(len(gt_idxs))
            seen += 1
            acc_seq_accuracy += seq_acc
            acc_gen_accuracy += gen_acc
            acc_data.append(dict(seq_accuracy=seq_acc, gen_accuracy=gen_acc))
            if verbose:
                print("   GT", "".join([id2a(c) for c in list(gt_seq.data.cpu())]))
                print("   PR", "".join([id2a(c) for c in list(pr_seq.data.cpu())]))
                print("   {:3d}: Seq {:.3f}  Generator {:.3f}  rmsd {:.3f}  min {:.3f}  median {:.3f}  max {:.3f}"
                      .format(seen, seq_acc, gen_acc, rmsd, min_dist, median_dist, max_dist))
    acc_seq_accuracy = acc_seq_accuracy / seen
    acc_seq_accuracy = acc_seq_accuracy / seen
    df_val = pd.DataFrame(acc_data)
    print(df_val.describe())


def predict(model1, model2, x, seq, params, device1, device2, conf_cutoff=0.1, ca_cutoff=4.0, verbose=False):
    x = torch.from_numpy(x.astype(np.float32))
    if verbose:
        print("x", x.shape)
    seq = [s for s in seq]
    z = torch.from_numpy(onehot(aa1toidx(seq)))
    if verbose:
        print("z", z.shape)
    matrix_path = "rotamer2aa_ukn.npy"
    rot2aa = torch.from_numpy(np.load(matrix_path, allow_pickle=True)).to(device2).float()
    t1 = time.time()
    with torch.no_grad():
        x = x.to(device1)
        z = z.to(device2)
        # model1.eval()
        model1.train()
        h = x.size(0)
        if h > 400:
            zi = ([0, 8*int(h/2/8)]), (8*int(h/2/8), h)
            results = list()
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # x_chunk = x[None, None, zi[i][0]:zi[i][1], zi[j][0]:zi[j][1], zi[k][0]:zi[k][1]]
                        # x_chunk.requires_grad = False
                        # x_chunk.to(device1)
                        output_chunk = model1(x[None, None, zi[i][0]:zi[i][1], zi[j][0]:zi[j][1], zi[k][0]:zi[k][1]])
                        print(i, j, k, output_chunk.size())
                        results.append(output_chunk)
            part_a = torch.cat((results[0], results[1]), 4)
            part_b = torch.cat((results[2], results[3]), 4)
            part_c = torch.cat((results[4], results[5]), 4)
            part_d = torch.cat((results[6], results[7]), 4)
            output1 = torch.cat((torch.cat((part_a, part_b), 3), torch.cat((part_c, part_d), 3)), 2)
            print("output1", output1.size())
        else:
            # x = x.to(device1)
            output1 = model1(x[None, None, :, :, :])
        ca_conf, ca_coors, cat_scores = extract(output1[0, :, :, :, :], cutoff=conf_cutoff)
        ca_conf.to(device2)
        ca_coors.to(device2)
        cat_scores.to(device2)
        # print("ca_conf", ca_conf.size())
        # print("ca_coors", ca_coors.size())
        # print("cat_scores", cat_scores.size())
        aa_scores = torch.max(torch.mm(torch.exp(cat_scores), rot2aa).view(-1, 21, 34), dim=2)[0][:, :-1]
        aa_scores = aa_scores / torch.sum(aa_scores, dim=1)[:, None]
        # print("aa_scores", aa_scores.size())
        ca_dists = torch.norm(ca_coors[:, None, :] - ca_coors[None, :, :], dim=2)
        # print("ca_dists", ca_dists.size())
        adj_mat = (ca_dists < ca_cutoff).float()
        # print("adj_mat", adj_mat.size())
        d_mat = torch.eye(adj_mat.size(0)).to(device2).float() * torch.pow(torch.sum(adj_mat, dim=1), -0.5).to(device2)
        # print("d_mat", d_mat.size())

        model2.eval()
        # print("model2", model2)
        ca_mask = torch.ones(ca_coors.size(0), ).to(device2) > 0
        # aa_scores.to(device2)
        # print("ca_mask", ca_mask.size())
        output2, idxs = model2(z.float(), aa_scores, adj_mat, d_mat, coor_tensor=ca_coors, mask_tensor=ca_mask)
        # print("running modedl2")
        # output2, idxs = model2(z.float(), aa_scores, adj_mat, d_mat, coor_tensor=None, mask_tensor=None)
        # print("output2", output2.size())
        # print("idxs", idxs.size())
        df_predict = dict()
        ca_coors_cpu = ca_coors[idxs, :].cpu().numpy()
        df_predict["CA_x"] = [coor[2] for coor in ca_coors_cpu]
        df_predict["CA_y"] = [coor[1] for coor in ca_coors_cpu]
        df_predict["CA_z"] = [coor[0] for coor in ca_coors_cpu]
        rot_labels = torch.argmax(cat_scores[idxs, :], 1)
        aa_labels = torch.argmax(aa_scores[idxs, :], 1)
        aa_gt = torch.argmax(z, 1)
        df_predict["rot_label"] = [label for label in rot_labels.cpu().numpy()]
        df_predict["aa_label"] = [label for label in aa_labels.cpu().numpy()]
        df_predict["aa_gg"] = [label for label in aa_gt.cpu().numpy()]
        df_predict = pd.DataFrame(df_predict)
        print(df_predict)
        df_predict.to_csv("output/predict.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_path", type=str, default="backup", help="Path to put output files")
    p.add_argument("--n_graph_layers", type=int, default=2, help="Number of layers in GCN")
    p.add_argument("--n_epoches", type=int, default=1, help="Number of training epoches")
    p.add_argument("--val_every", type=int, default=1024, help="Validate every X steps")
    p.add_argument("--weight_decay", type=float, default=0.0, help="L2 weight decay")
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    p.add_argument("--gpu", "-g", type=int, default=None, help="Use GPU x")
    p.add_argument("--gpu2", type=int, default=None, help="Use GPU y for the LSTM model")
    p.add_argument("--df_train", type=str, default=None, help="Path to training set DF")
    p.add_argument("--df_val", type=str, default=None, help="Path to validation set DF")
    p.add_argument("--map", type=str, default=None, help="Path to the h5 file converted from map")
    p.add_argument("--fasta", type=str, default=None, help="Path to the fasta file")
    p.add_argument("--result_path", type=str, default=None, help="Path to AlphaNet outputs")
    p.add_argument("--n_lstm_hidden", type=int, default=256, help="Dimenion of LSTM hidden states")
    p.add_argument("--n_node_embed", type=int, default=128, help="Dimenion of node embeddings")
    p.add_argument("--n_seq_embed", type=int, default=32, help="Dimenion of sequence embeddings")
    p.add_argument("--n_dist_embed", type=int, default=0, help="Dimenion of distance embeddings")
    p.add_argument("--model1", "-a", type=str, default=None, help="Path to the pretrained AlphaNet model")
    p.add_argument("--model2", "-m", type=str, default=None, help="Path to the pretrained Generator model")
    p.add_argument("--conf_cutoff", "-d", type=float, default=0.1, help="AlphaNet confidence cutoff")
    p.add_argument("--ca_cutoff", "-c", type=float, default=4.0, help="C-alpha contact cutoff in Ang")
    p.add_argument("--dist_loss_weight", type=float, default=0.0, help="Weight for CA dist loss")
    p.add_argument("--max_size", type=int, default=-1, help="Maximum volume size")
    p.add_argument("--max_dist", type=float, default=None, help="Maximum distance from ground truth coordinates")
    p.add_argument("--seed", type=int, default=2020, help="Random seed for NumPy and Pandas")
    p.add_argument("--query", "-q", type=str, default=None, help="Query for selecting only a subset of data")
    p.add_argument("--use_tp", action="store_true", help="Use true positive nodes")
    p.add_argument("--nms", type=float, default=-0.01, help="Non-Maximum suppression threshold (in Ang)")
    p.add_argument("--skip", action="store_true", help="Skip training")
    p.add_argument("--verbose", "-v", action="store_true", help="Be verbose")
    return p.parse_args()

    
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model1 = AlphaNet(n_filters=32, bottleneck=False)
    model2 = GeneratorLSTM(n_graph_layers=args.n_graph_layers, device=args.gpu, n_lstm_hidden=args.n_lstm_hidden,
                           n_node_embed=args.n_node_embed, n_seq_embed=args.n_seq_embed, graph_to_lstm=False,
                           bidirectional_lstm=True, n_distance_feat=args.n_dist_embed)
    model2.seen = 0
    if args.model1 is not None:
        model1.load_state_dict(torch.load(args.model1, map_location="cpu"))
        print("trained model {} loaded".format(args.model1))
    if args.model2 is not None:
        model2.load_state_dict(torch.load(args.model2, map_location="cpu"))
        print("trained model {} loaded".format(args.model2))
    if args.gpu is not None and torch.cuda.is_available():
        device1 = torch.device("cuda:{}".format(args.gpu))
        if args.gpu2 is not None:
            device2 = torch.device("cuda:{}".format(args.gpu2))
        else:
            device2 = device1
    else:
        device1 = torch.device("cpu")
        device2 = torch.device("cpu")
    model1 = model1.to(device1)
    model2 = model2.to(device2)
    if args.verbose:
        print("AlphaNet")
        print(model1)
        print("Generator")
        print(model2)
    params = dict(epochs=args.n_epoches, batch_size=1, num_workers=0, lr=args.lr, print_every=1, back_every=1,
                  save_every=128, val_every=args.val_every, weight_decay=args.weight_decay)
    if args.map is not None:
        seq = open(args.fasta, "r").read().splitlines()[1]
        print(seq)
        with h5py.File(args.map, "r") as f:
            img = f["data"][()]
        print("img", img.shape)
        predict(model1, model2, img, seq, params, device1, device2, conf_cutoff=args.conf_cutoff,
                ca_cutoff=args.ca_cutoff, verbose=True)
    else:
        if args.skip:
            pass
        else:
            os.makedirs(args.output_path, exist_ok=True)
            train_dataset = AlphaSeqDataset(args.df_train, normalize=True, transform=False, downsample=8, n=-1,
                                            max_size=args.max_size, query="n_alphabet<=20 and n_chains == 1",
                                            seed=args.seed)
            model2 = train(model1, model2, train_dataset, params, device1, device2, conf_cutoff=args.conf_cutoff,
                           ca_cutoff=args.ca_cutoff, dist_loss_weight=args.dist_loss_weight, verbose=True)
            output_filename = os.path.join(args.output_path, "ep_01.pt")
            torch.save(model2.state_dict(), output_filename)
        if args.df_val is not None:
            val_dataset = AlphaSeqDataset(args.df_val, normalize=True, transform=False, downsample=8, n=-1,
                                          max_size=args.max_size, query=args.query, seed=args.seed)
            print("val dataset", len(val_dataset))
            val(model1, model2, val_dataset, params, device1, device2, conf_cutoff=args.conf_cutoff,
                ca_cutoff=args.ca_cutoff, use_tp=args.use_tp, nms=args.nms, verbose=True)


if __name__ == "__main__":
    main()
