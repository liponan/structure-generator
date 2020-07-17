import torch
import torch.nn as nn


class GCN(nn.Module):

    def __init__(self, n_features, n_out, n_hidden=None, n_layers=1, bias=False):
        super(GCN, self).__init__()
        self.n_layers = n_layers
        print("GCN layers: {}".format(self.n_layers))
        if n_hidden is None:
            n_hidden = n_out
        if n_layers == 1:
            self.W_out = nn.Linear(n_features, n_out, bias=bias)
        else:
            self.W1 = nn.Linear(n_features, n_hidden, bias=bias)
            if n_layers > 2:
                self.W2 = nn.Linear(n_hidden, n_hidden, bias=bias)
            if n_layers > 3:
                self.W3 = nn.Linear(n_hidden, n_hidden, bias=bias)
            if n_layers > 4:
                self.W4 = nn.Linear(n_hidden, n_hidden, bias=bias)
            self.W_out = nn.Linear(n_hidden, n_out, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, mat_a, mat_d, mat_f):
        mat_d = torch.mm(torch.mm(mat_d, mat_a), mat_d)
        x = mat_f
        if self.n_layers > 1:
            x = torch.mm(mat_d, x)
            x = self.relu(self.W1(x))
        if self.n_layers > 2:
            x = torch.mm(mat_d, x)
            x = self.relu(self.W2(x))
        if self.n_layers > 3:
            x = torch.mm(mat_d, x)
            x = self.relu(self.W3(x))
        if self.n_layers > 4:
            x = torch.mm(mat_d, x)
            x = self.relu(self.W4(x))
        # output layer
        if self.n_layers > 0:
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
        self.gcn = GCN(n_node_embed, n_node_embed, n_node_embed, n_graph_layers, bias=True)
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


class FocalLoss(nn.Module):

    def __init__(self, weight, gamma=2, reduce=True, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduce = reduce
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none", ignore_index=ignore_index)
        self.mod = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    def forward(self, scores, targets):
        ce_loss = self.ce(scores, targets)
        f_loss = (1 - torch.exp(-self.mod(scores, targets)))**self.gamma * ce_loss
        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss
