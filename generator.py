import os
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
import argparse


def train(model, dataset, val_dataset=None, n_epoch=1, lr=0.1, print_every=100, log_every=100, val_every=2000,
          focalloss=None, device=None, verbose=False):
    print("====================== train ======================")
    t0 = time.time()
    log = dict(train=list(), val=list(), val_seen=list())
    if focalloss is not None:
        loss_fn = FocalLoss(weight=None, gamma=focalloss)
    else:
        loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    for i in range(n_epoch):
        t1 = time.time()
        for j, data in enumerate(dataloader):
            model.train()
            model.zero_grad()
            x, f, a_mat, d_mat, gt_idxs = data
            if verbose:
                print("x", x.size(), "f", f.size(), "a_mat", a_mat.size(), "d_mat", d_mat.size(),
                      "gt_idxs", gt_idxs.size())
            n = len(gt_idxs[0, :])
            if verbose:
                print(j, list(gt_idxs[0, :]), n)
            x_tensor = x[0, :].float()
            f_tensor = f[0, :].float()
            a_tensor = a_mat[0, :].float()
            d_tensor = d_mat[0, :].float()
            g_tensor = gt_idxs[0, :].long()
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


def val(model, dataset, device=None, verbose=False, reverse_seq=False, output=None):
    acc_all = 0
    model.eval()
    with torch.no_grad():
        for j in range(len(dataset)):
            model.zero_grad()
            x, f, a_mat, d_mat, gt_idxs = dataset[j]
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
            scores, idxs = model(x_tensor, f_tensor, a_tensor, d_tensor)
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
            if output is not None:
                scores_np = scores.data.cpu().numpy()
                np.save(os.path.join(output, "{}.npy".format(str(j).zfill(6))), scores_np)
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
    p.add_argument("--focalloss", type=float, default=None, help="Gamma parameter for FocalLoss")
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
    p.add_argument("--save_val", type=str, default=None, help="Path saving inference results")
    p.add_argument("--model", "-m", type=str, default=None, help="Path to the pretrained model")
    p.add_argument("--log", type=str, default=None, help="Path and file name for saving training log")
    p.add_argument("--skip_training", action="store_true", help="Skip training")
    p.add_argument("--reverse_seq", action="store_true", help="Reverse sequence when validating")
    p.add_argument("--h5_tmp", type=str, default=None, help="Save simulated data as h5 file to the given path")
    p.add_argument("--build_on_the_fly", action="store_true", help="Generate simulated data on the fly")
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
        train_dataset = GCNDataset(n=args.n_train, max_buf_size=args.max_n_seq, df_path=args.df, h5=args.h5_tmp,
                                   seq_len_range=(args.min_len, args.max_len), build_on_the_fly=args.build_on_the_fly,
                                   seed=args.seed, verbose=False)
        model, json_log = train(model, train_dataset, val_dataset=val_dataset, n_epoch=args.n_epoch, lr=args.lr,
                                device=device, focalloss=args.focalloss, verbose=args.verbose)
        if args.save:
            torch.save(model.state_dict(), args.save)
        if args.log:
            json_log["params"] = args.__dict__
            with open(args.log, "w") as f:
                f.write(json.dumps(json_log))
    val_acc = val(model, val_dataset, device=device, verbose=True, reverse_seq=args.reverse_seq, output=args.save_val)
    print("test on validation set: {:.5f}".format(val_acc))


if __name__ == "__main__":
    main()
