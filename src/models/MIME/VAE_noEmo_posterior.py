import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import config


class VAESampling(nn.Module):
    def __init__(self, hidden_dim, posterior_hidden_dim, out_dim=32):
        super().__init__()
        self.positive_emotions = [11, 16, 6, 8, 3, 1, 28, 13, 31, 17, 24, 0, 27]
        self.negative_emotions = [
            9,
            4,
            2,
            14,
            30,
            29,
            25,
            15,
            10,
            23,
            19,
            18,
            21,
            7,
            20,
            5,
            26,
            12,
            22,
        ]  # anticipation is negative
        self.positive_emotions_t = torch.LongTensor(self.positive_emotions).to(
            config.device
        )
        self.negative_emotions_t = torch.LongTensor(self.negative_emotions).to(
            config.device
        )
        # Prior encoding
        self.h_prior = nn.Linear(hidden_dim, hidden_dim)

        self.mu_prior_positive = nn.Linear(hidden_dim, out_dim)
        self.logvar_prior_positive = nn.Linear(hidden_dim, out_dim)
        self.Dense_z_prior_positive = nn.Linear(out_dim, len(self.positive_emotions))

        self.mu_prior_negative = nn.Linear(hidden_dim, out_dim)
        self.logvar_prior_negative = nn.Linear(hidden_dim, out_dim)
        self.Dense_z_prior_negative = nn.Linear(out_dim, len(self.negative_emotions))

        # Posterior encoder
        self.h_posterior_postive = nn.Linear(
            hidden_dim + posterior_hidden_dim, hidden_dim
        )
        self.h_posterior_negative = nn.Linear(
            hidden_dim + posterior_hidden_dim, hidden_dim
        )

        self.mu_posterior_positive = nn.Linear(hidden_dim, out_dim)
        self.logvar_posterior_positive = nn.Linear(hidden_dim, out_dim)
        self.Dense_z_posterior_positive = nn.Linear(
            out_dim, len(self.positive_emotions)
        )

        self.mu_posterior_negative = nn.Linear(hidden_dim, out_dim)
        self.logvar_posterior_negative = nn.Linear(hidden_dim, out_dim)
        self.Dense_z_posterior_negative = nn.Linear(
            out_dim, len(self.negative_emotions)
        )

    def prior(self, x):
        h1 = F.relu(self.h_prior(x))
        mu_positive = self.mu_prior_positive(h1)
        logvar_positive = self.logvar_prior_positive(h1)
        mu_negative = self.mu_prior_negative(h1)
        logvar_negative = self.logvar_prior_negative(h1)
        return mu_positive, logvar_positive, mu_negative, logvar_positive

    def posterior(self, x, e, M_out, M_tilde_out):
        h1_positive = torch.zeros(M_out.shape).to(config.device)
        h1_negative = torch.zeros(M_out.shape).to(config.device)
        for i in range(len(e)):
            if self.is_pos(e[i]):
                h1_positive[i] = M_out[i]
                h1_negative[i] = M_tilde_out[i]
            else:
                h1_positive[i] = M_tilde_out[i]
                h1_negative[i] = M_out[i]
        # Postive
        x_positive = torch.cat([x, h1_positive], dim=-1)
        h1_positive = F.relu(self.h_posterior_postive(x_positive))
        mu_positive = self.mu_posterior_positive(h1_positive)
        logvar_positive = self.logvar_posterior_positive(h1_positive)
        # Negative
        x_negative = torch.cat([x, h1_negative], dim=-1)
        h1_negative = F.relu(self.h_posterior_negative(x_negative))
        mu_negative = self.mu_posterior_negative(h1_negative)
        logvar_negative = self.logvar_posterior_negative(h1_negative)

        return mu_positive, logvar_positive, mu_negative, logvar_positive

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def is_pos(self, e):
        return e in self.positive_emotions

    def forward(self, q_h, e, emb_layer):
        """This method is for evaluation only"""
        x = q_h
        mu_p, logvar_p, mu_n, logvar_n = self.prior(x)

        z_p = self.reparameterize(mu_p, logvar_p)
        E_prob_p = torch.softmax(
            self.Dense_z_prior_positive(z_p), dim=-1
        )  # (bs, len(pos))
        emotions_p = E_prob_p @ emb_layer(self.positive_emotions_t)  # (bs, dim)

        z_n = self.reparameterize(mu_n, logvar_n)
        E_prob_n = torch.softmax(
            self.Dense_z_prior_negative(z_n), dim=-1
        )  # (bs, len(neg))
        emotions_n = E_prob_n @ emb_layer(self.negative_emotions_t)

        emotions_mimic = torch.zeros(emotions_n.shape)
        emotions_non_mimic = torch.zeros(emotions_n.shape)
        for i in range(len(e)):
            if self.is_pos(e[i]):
                emotions_mimic[i] = emotions_p[i]
                emotions_non_mimic[i] = emotions_n[i]
            else:
                emotions_mimic[i] = emotions_n[i]
                emotions_non_mimic[i] = emotions_p[i]

        emotions_mimic.to(config.device)
        emotions_non_mimic.to(config.device)

        return emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n

    def forward_train(self, q_h, e, emb_layer, M_out, M_tilde_out):
        mu_p, logvar_p, mu_n, logvar_n = self.posterior(q_h, e, M_out, M_tilde_out)

        z_p = self.reparameterize(mu_p, logvar_p)
        E_prob_p = torch.softmax(
            self.Dense_z_prior_positive(z_p), dim=-1
        )  # (bs, len(pos))
        emotions_p = E_prob_p @ emb_layer(self.positive_emotions_t)  # (bs, dim)

        z_n = self.reparameterize(mu_n, logvar_n)
        E_prob_n = torch.softmax(
            self.Dense_z_prior_negative(z_n), dim=-1
        )  # (bs, len(neg))
        emotions_n = E_prob_n @ emb_layer(self.negative_emotions_t)

        emotions_mimic = torch.zeros(emotions_n.shape)
        emotions_non_mimic = torch.zeros(emotions_n.shape)
        for i in range(len(e)):
            if self.is_pos(e[i]):
                emotions_mimic[i] = emotions_p[i]
                emotions_non_mimic[i] = emotions_n[i]
            else:
                emotions_mimic[i] = emotions_n[i]
                emotions_non_mimic[i] = emotions_p[i]
        emotions_mimic.to(config.device)
        emotions_non_mimic.to(config.device)

        return emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n

    @staticmethod
    def kl_div(mu_posterior, logvar_posterior, mu_prior=None, logvar_prior=None):
        """
        This code is adapted from:
        https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling/blob/83ca9dd96272d3a38978a1dfa316d06d5d6a7c77/model/utils/probability.py#L20
        """
        one = torch.FloatTensor([1.0]).to(config.device)
        if mu_prior == None:
            mu_prior = torch.FloatTensor([0.0]).to(config.device)
            logvar_prior = torch.FloatTensor([0.0]).to(config.device)
        kl_div = torch.sum(
            0.5
            * (
                logvar_prior
                - logvar_posterior
                + (logvar_posterior.exp() + (mu_posterior - mu_prior).pow(2))
                / logvar_prior.exp()
                - one
            )
        )
        return kl_div
