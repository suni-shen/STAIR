import torch
import torch.nn as nn
import torch.nn.functional as F
TRANSFER_CFG = {'sigma_x': 'auto', 'sigma_y': 1.0, 'lambda_jmmd': 1.0, 'lambda_disc': 0.1}

@torch.no_grad()
def _median_sigma(X, Y=None):
    if Y is not None:
        Z = torch.cat([X, Y], dim=0)
    else:
        Z = X
    ZZ = Z.pow(2).sum(-1, keepdim=True)
    dist_sq = ZZ + ZZ.t() - 2.0 * Z @ Z.t()
    mask = torch.triu(torch.ones_like(dist_sq, dtype=torch.bool), diagonal=1)
    dists = dist_sq[mask].clamp(min=1e-10).sqrt()
    sigma = dists.median()
    return sigma.clamp(min=1e-05)

class LowRankProjector(nn.Module):

    def __init__(self, d_in: int, d_out: int, rank: int):
        super().__init__()
        r = min(rank, d_in, d_out)
        self.B = nn.Linear(d_in, r, bias=False)
        self.A = nn.Linear(r, d_out, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.B.weight)
        nn.init.orthogonal_(self.A.weight)

    def forward(self, x):
        return self.A(self.B(x))

def build_projectors(num_concat, d_feat, rank):
    return nn.ModuleList([LowRankProjector(d_i, d_feat, rank) for d_i in num_concat])

def _rbf_kernel(X, Y, sigma):
    XX = X.pow(2).sum(-1, keepdim=True)
    YY = Y.pow(2).sum(-1, keepdim=True)
    dist_sq = XX + YY.t() - 2.0 * X @ Y.t()
    return (-dist_sq / (2.0 * sigma ** 2)).exp()

def _label_kernel(y, is_cls, sigma_y):
    if is_cls:
        return (y.unsqueeze(0) == y.unsqueeze(1)).float()
    if sigma_y is None or sigma_y == 'auto':
        sigma_y = _median_sigma(y.unsqueeze(-1))
    diff_sq = (y.unsqueeze(0) - y.unsqueeze(1)).pow(2)
    return (-diff_sq / (2.0 * sigma_y ** 2)).exp()

def _center_kernel(K):
    row_mean = K.mean(dim=1, keepdim=True)
    col_mean = K.mean(dim=0, keepdim=True)
    return K - row_mean - col_mean + K.mean()

def _hsic(K_x, K_y):
    B = K_x.size(0)
    return (_center_kernel(K_x) * _center_kernel(K_y)).sum() / (B - 1) ** 2

def _expert_jmmd(p_i, z_i, K_y, sigma_x):
    if sigma_x is None or sigma_x == 'auto':
        sigma_x = _median_sigma(p_i, z_i)
    K_pp = _rbf_kernel(p_i, p_i, sigma_x)
    K_zz = _rbf_kernel(z_i, z_i, sigma_x)
    K_pz = _rbf_kernel(p_i, z_i, sigma_x)
    return ((K_pp + K_zz - 2.0 * K_pz) * K_y).mean()

def jmmd_hsic_transfer_loss(segs, feats, projectors, y, beta, is_cls, cfg=None):
    cfg = {**TRANSFER_CFG, **(cfg or {})}
    (sigma_x, sigma_y) = (cfg['sigma_x'], cfg['sigma_y'])
    (lam_jmmd, lam_disc) = (cfg.get('lambda_jmmd', 1.0), cfg['lambda_disc'])
    M = len(segs)
    beta_bar = beta.detach().mean(dim=0)
    K_y = _label_kernel(y, is_cls, sigma_y)
    l_transfer = torch.tensor(0.0, device=y.device)
    for i in range(M):
        e_i = segs[i].detach()
        p_i = projectors[i](e_i)
        z_i = feats[i]
        l_jmmd_i = _expert_jmmd(p_i, z_i, K_y, sigma_x)
        sig_z = _median_sigma(z_i)
        K_z = _rbf_kernel(z_i, z_i, sig_z)
        l_disc_i = -_hsic(K_z, K_y)
        l_transfer = l_transfer + beta_bar[i] * (lam_jmmd * l_jmmd_i + lam_disc * l_disc_i)
    return l_transfer


def task_loss(prediction, target, classification):
    if classification:
        return F.cross_entropy(prediction, target)
    return F.mse_loss(prediction.squeeze(-1), target)


def stair_loss(prediction, target, mask, aux, projectors, training, classification):
    diagonal = aux["y_diag"]
    if classification:
        batch, sources, classes = diagonal.shape
        labels = target.unsqueeze(1).expand(batch, sources).reshape(-1)
        individual = F.cross_entropy(diagonal.reshape(-1, classes), labels, reduction="none")
        individual = individual.reshape(batch, sources)
    else:
        individual = (diagonal.squeeze(-1) - target.unsqueeze(1)).pow(2)
    selected = mask.detach().float()
    exploitation = (selected * individual).sum() / (selected.sum() + 1e-8)
    transfer = jmmd_hsic_transfer_loss(
        aux["segs"], aux["feats"], projectors, target, aux["w_soft"],
        classification, training["transfer_cfg"],
    )
    return (task_loss(prediction, target, classification)
            + training["l_exploit"] * exploitation + training["l_transfer"] * transfer)
