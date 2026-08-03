"""
This function is adapted from [OmniAnomaly] by [TsingHuasuya et al.]
Original source: [https://github.com/NetManAIOps/OmniAnomaly]
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import torch.nn.functional as F
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import tqdm

from .base import BaseDetector
from ..utils.dataset import ReconstructDataset
from ..utils.torch_utility import EarlyStoppingTorch, get_gpu

class PlanarFlow(nn.Module):
    """Single planar normalizing flow: z' = z + u * tanh(w^T z + b)."""
    def __init__(self, n_latent):
        super(PlanarFlow, self).__init__()
        self.w = nn.Parameter(torch.empty(n_latent).normal_(0, 0.01))
        self.b = nn.Parameter(torch.zeros(1))
        self.u = nn.Parameter(torch.empty(n_latent).normal_(0, 0.01))

    def forward(self, z):
        # z: (seq, batch, n_latent)
        lin = (z * self.w).sum(-1, keepdim=True) + self.b          # (seq, batch, 1)
        # Enforce invertibility: w^T u_hat >= -1
        wu = (self.w * self.u).sum()
        m_wu = -1.0 + F.softplus(wu)
        u_hat = self.u + (m_wu - wu) * self.w / ((self.w * self.w).sum() + 1e-8)
        z_new = z + u_hat * torch.tanh(lin)
        psi = (1.0 - torch.tanh(lin) ** 2) * self.w                # (seq, batch, n_latent)
        log_det = torch.log(torch.abs(1.0 + (psi * u_hat).sum(-1, keepdim=True)) + 1e-8)
        return z_new, log_det                                        # log_det: (seq, batch, 1)


class OmniAnomalyModel(nn.Module):
    def __init__(self, feats, device, n_flows=20):
        super(OmniAnomalyModel, self).__init__()
        self.name = 'OmniAnomaly'
        self.device = device
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        # Encoder RNN + MLP (outputs mu and log_std, not log_var)
        self.encoder_rnn = nn.GRU(feats, self.n_hidden, 2)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, 2 * self.n_latent)
        )
        # Planar normalizing flows on the latent sample
        self.flows = nn.ModuleList([PlanarFlow(self.n_latent) for _ in range(n_flows)])
        # Decoder RNN + MLP (no Sigmoid — data need not be in [0,1])
        self.decoder_rnn = nn.GRU(self.n_latent, self.n_hidden, 2)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats)
        )

    def forward(self, x, hidden=None):
        # x: (batch, win, feats)
        bs, win, _ = x.shape
        x_seq = x.view(win, bs, self.n_feats)           # (win, bs, feats)

        # Encode
        enc_out, hidden = self.encoder_rnn(x_seq, hidden)   # enc_out: (win, bs, n_hidden)
        enc = self.encoder_mlp(enc_out)                      # (win, bs, 2*n_latent)
        mu, log_std = torch.split(enc, self.n_latent, dim=-1)
        std = torch.exp(log_std)
        z = mu + torch.randn_like(std) * std                 # (win, bs, n_latent)

        # Planar normalizing flows
        log_det_sum = z.new_zeros(win, bs, 1)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum = log_det_sum + log_det

        # Decode (RNN then MLP — no Sigmoid)
        dec_out, _ = self.decoder_rnn(z)                     # (win, bs, n_hidden)
        x_hat = self.decoder_mlp(dec_out)                    # (win, bs, n_feats)

        # Permute back to (bs, win, feats / n_latent) for downstream use
        x_hat   = x_hat.permute(1, 0, 2)                    # (bs, win, feats)
        mu      = mu.permute(1, 0, 2)                        # (bs, win, n_latent)
        log_std = log_std.permute(1, 0, 2)                   # (bs, win, n_latent)
        log_det_sum = log_det_sum.squeeze(-1).permute(1, 0)  # (bs, win)

        return (
            x_hat.reshape(bs, win * self.n_feats),
            mu.reshape(bs, win * self.n_latent),
            log_std.reshape(bs, win * self.n_latent),
            log_det_sum,
            hidden,
        )


class OmniAnomaly(BaseDetector):
    def __init__(self,
                 win_size = 5,
                 feats = 1,
                 batch_size = 128,
                 epochs = 50,
                 patience = 3,
                 lr = 0.002,
                 validation_size=0.2
                 ):
        super().__init__()

        self.__anomaly_score = None

        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = win_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.validation_size = validation_size

        self.model = OmniAnomalyModel(feats=self.feats, device=self.device).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        self.criterion = nn.MSELoss(reduction = 'none')

        self.early_stopping = EarlyStoppingTorch(None, patience=patience)

    def fit(self, data):
        tsTrain = data[:int((1-self.validation_size)*len(data))]
        tsValid = data[int((1-self.validation_size)*len(data)):]

        train_loader = DataLoader(
            dataset=ReconstructDataset(tsTrain, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            dataset=ReconstructDataset(tsValid, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        mses, klds = [], []
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            n = epoch + 1
            avg_loss = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), leave=True
            )
            for idx, (d, _) in loop:
                d = d.to(self.device)

                y_pred, mu, log_std, log_det_sum, hidden = self.model(d, None)
                hidden = hidden.detach()
                d_flat = d.view(-1, self.feats * self.win_size)
                # Reconstruction NLL: -log p(x|z) = 0.5 * sum((x - x_hat)^2)
                recon = 0.5 * torch.sum(self.criterion(y_pred, d_flat), dim=-1)
                # KL with normalizing-flow correction: KL[q(z0)||p(z)] - sum_k log|det J_k|
                std = torch.exp(log_std)
                kl_gauss = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 1.0 - 2.0 * log_std, dim=-1)
                KLD = kl_gauss - log_det_sum.sum(dim=-1)
                loss = torch.mean(recon + self.model.beta * KLD)

                mses.append(torch.mean(recon).item())
                klds.append(self.model.beta * torch.mean(KLD).item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.cpu().item()
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            if len(valid_loader) > 0:
                self.model.eval()
                avg_loss_val = 0
                loop = tqdm.tqdm(
                    enumerate(valid_loader), total=len(valid_loader), leave=True
                )
                with torch.no_grad():
                    for idx, (d, _) in loop:
                        d = d.to(self.device)
                        y_pred, mu, log_std, log_det_sum, hidden = self.model(d, None)
                        hidden = hidden.detach()
                        d_flat = d.view(-1, self.feats * self.win_size)
                        recon = 0.5 * torch.sum(self.criterion(y_pred, d_flat), dim=-1)
                        std = torch.exp(log_std)
                        kl_gauss = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 1.0 - 2.0 * log_std, dim=-1)
                        KLD = kl_gauss - log_det_sum.sum(dim=-1)
                        loss = torch.mean(recon + self.model.beta * KLD)

                        avg_loss_val += loss.cpu().item()
                        loop.set_description(
                            f"Validation Epoch [{epoch}/{self.epochs}]"
                        )
                        loop.set_postfix(loss=loss.item(), avg_loss_val=avg_loss_val / (idx + 1))

            self.scheduler.step()
            if len(valid_loader) > 0:
                avg_loss = avg_loss_val / len(valid_loader)
            else:
                avg_loss = avg_loss / len(train_loader)
            self.early_stopping(avg_loss, self.model)
            if self.early_stopping.early_stop:
                # print("   Early stopping<<<")
                break

    def decision_function(self, data):
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size),
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model.eval()
        scores = []
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

        with torch.no_grad():
            for _, (d, _) in loop:
                d = d.to(self.device)

                y_pred, mu, log_std, log_det_sum, hidden = self.model(d, None)
                hidden = hidden.detach()
                bs = d.shape[0]

                # Score = negative ELBO on the last time-step
                y_pred_last = y_pred.view(bs, self.win_size, self.feats)[:, -1, :]  # (bs, feats)
                d_last      = d[:, -1, :]                                            # (bs, feats)
                recon_last  = 0.5 * torch.sum(self.criterion(y_pred_last, d_last), dim=-1)

                std = torch.exp(log_std)
                kl_gauss = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 1.0 - 2.0 * log_std, dim=-1)
                KLD = kl_gauss - log_det_sum.sum(dim=-1)
                score = recon_last + self.model.beta * KLD

                scores.append(score.cpu())
        
        scores = torch.cat(scores, dim=0)
        scores = scores.numpy()

        self.__anomaly_score = scores

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.win_size-1)/2) + 
                        list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.win_size-1)//2))
        
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass
