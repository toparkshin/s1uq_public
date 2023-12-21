import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.config import Config


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
                0.5
                * x
                * (
                        1.0
                        + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
                )
        )


class SelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # mult by 3 for q,k,v
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.n_embd = config.n_embd

    def forward(self, x):
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # self-attention
        att = (q @ k.transpose(1, 2)) * (1.0 / math.sqrt(k.size(-1)))  # (B, N, N)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, N, L)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(
                    config.resid_pdrop,
                ),  # TODO: this assumes i.i.d. for all values. nn.Dropout1d assumes indepdnence among channels
            ),
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlpf(self.ln2(x))
        return x


class SignalEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_proj = nn.Linear(config.n_signal, config.n_embd)

    def forward(self, x):
        x = self.c_proj(x)
        return x


class UncertaintyHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.include_aleatoric = (
                config.uq_mode == "combined" or config.uq_mode == "aleatoric"
        )
        if self.include_aleatoric:
            self.head = nn.Linear(config.n_embd, config.n_class * 2, bias=False)
        else:
            assert config.uq_mode == "epistemic"
            self.head = nn.Linear(config.n_embd, config.n_class, bias=False)

    def _sample_mu(self, mu, sigma):
        N = mu.shape[0]

        assert (
                tuple(mu.shape) == tuple(sigma.shape) == (N, self.config.n_class)
        ), f"mu.shape: {tuple(mu.shape)}\nsigma.shape{tuple(sigma.shape)}\n{(N, self.config.n_class)}"

        probs = torch.empty(
            (
                self.config.n_logit_samples,
                N,
                self.config.n_class,
            ),
            device=mu.device,
        )
        for t in range(self.config.n_logit_samples):
            eps = torch.randn(sigma.shape, device=sigma.device)
            logit = mu + torch.mul(sigma, eps)
            logit_max = torch.max(logit, dim=1, keepdims=True).values
            probs[t] = F.softmax(
                logit - logit_max, dim=1
            )  # for numerical stability

        prob = torch.mean(probs, dim=0)
        return torch.exp(prob)

    def forward(self, x, training=False):
        x = self.head(x)

        if self.include_aleatoric:
            mu, logvar = x.split(self.config.n_class, dim=1)
            sigma = torch.sqrt(torch.exp(logvar))
            if training:  # for training
                mu = self._sample_mu(mu, sigma)
            return mu, sigma
        else:
            return x, None  # no sigma for epistemic only


class NeuronTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                signal_encoder=SignalEncoder(config),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln=nn.LayerNorm(config.n_embd),
            ),
        )
        self.head = UncertaintyHead(config)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    def forward(self, x, training=False):
        x = self.transformer.signal_encoder(x)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln(x)
        x = torch.max(x, dim=1).values
        mu, sigma = self.head(x, training=training)
        return mu, sigma

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
