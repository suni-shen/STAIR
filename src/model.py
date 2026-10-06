import torch
from torch import nn
from torch.nn import functional as F

from losses import build_projectors


class PredictionHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dp = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.dp(self.act(self.fc1(x)))
        return self.fc2(h), h


class STAIR(nn.Module):
    def __init__(self, source_dims, hidden_dim=64, output_dim=1, topk=6):
        super().__init__()
        if not 1 <= topk <= len(source_dims):
            raise ValueError("topk must be between 1 and the number of sources")
        self.source_dims = source_dims
        self.topk = topk
        self.normalize = nn.BatchNorm1d(sum(source_dims))
        self.projectors = build_projectors(source_dims, hidden_dim, 64)
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
            for dim in source_dims
        ])
        self.heads = nn.ModuleList([
            PredictionHead(hidden_dim, output_dim) for _ in source_dims
        ])
        self.WF = nn.Linear(hidden_dim, hidden_dim)
        self.WO = nn.Linear(hidden_dim, hidden_dim)
        self.WQ = nn.Linear(len(source_dims) * hidden_dim, hidden_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for adapter in self.adapters:
            nn.init.kaiming_uniform_(adapter[0].weight, nonlinearity="relu")
            nn.init.zeros_(adapter[0].bias)
            nn.init.ones_(adapter[1].weight)
            nn.init.zeros_(adapter[1].bias)
        for head in self.heads:
            nn.init.kaiming_uniform_(head.fc1.weight, nonlinearity="relu")
            nn.init.zeros_(head.fc1.bias)
            nn.init.normal_(head.fc2.weight, std=1e-3)
            nn.init.zeros_(head.fc2.bias)
        for layer in (self.WF, self.WO, self.WQ):
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.ones_(self.normalize.weight)
        nn.init.zeros_(self.normalize.bias)

    def forward(self, protein, epoch=0, epochs=200):
        segments = list(self.normalize(protein).split(self.source_dims, dim=-1))
        features = [adapter(x) for adapter, x in zip(self.adapters, segments)]
        assembled = []
        initial_phase = self.training and epoch <= epochs // 10
        for i, feature in enumerate(features):
            if initial_phase:
                assembled.append(self.heads[i](feature)[0])
            else:
                outputs = [head(feature) for head in self.heads]
                predictions = torch.stack([output[0] for output in outputs], dim=1)
                semantics = torch.stack([output[1] for output in outputs], dim=1)
                f = F.normalize(self.WF(feature), dim=-1)
                h = F.normalize(self.WO(semantics), dim=-1)
                alpha = F.softmax((h * f.unsqueeze(1)).sum(-1) / 0.5, dim=-1)
                assembled.append((alpha.unsqueeze(-1) * predictions).sum(dim=1))
        assembled = torch.stack(assembled, dim=1)
        query = F.normalize(self.WQ(torch.cat(features, dim=-1)), dim=-1)
        keys = F.normalize(torch.stack([self.WF(x) for x in features], dim=1), dim=-1)
        dense_weights = F.softmax((keys * query.unsqueeze(1)).sum(-1), dim=-1)
        weights = dense_weights
        mask = torch.ones_like(weights)
        if self.topk < len(features):
            indices = weights.topk(self.topk, dim=-1).indices
            mask = torch.zeros_like(weights).scatter_(1, indices, 1.0)
            sparse_weights = weights * mask
            sparse_weights = sparse_weights / (sparse_weights.sum(-1, keepdim=True) + 1e-8)
            weights = sparse_weights.detach() + (weights - weights.detach())
        prediction = (weights.unsqueeze(-1) * assembled).sum(dim=1)
        diagonal = torch.stack([
            head(feature)[0] for head, feature in zip(self.heads, features)
        ], dim=1)
        return prediction, mask, {
            "segs": segments,
            "feats": features,
            "w_soft": dense_weights,
            "w_final": weights,
            "y_diag": diagonal,
        }

    def parameter_groups(self, training):
        routing = [p for layer in (self.WF, self.WO, self.WQ) for p in layer.parameters()]
        experts = [p for layer in (self.adapters, self.heads, self.normalize, self.projectors)
                   for p in layer.parameters()]
        return [
            {"params": routing, "lr": training["lr_gate"]},
            {"params": experts, "lr": training["lr_expert"]},
        ]
