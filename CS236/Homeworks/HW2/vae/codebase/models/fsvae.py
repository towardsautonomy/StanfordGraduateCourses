# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
from numpy.core.fromnumeric import var
import torch
import torch.utils.data
from codebase import utils as ut
from codebase.models import nns
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class FSVAE(nn.Module):
    def __init__(self, nn='v2', name='fsvae'):
        super().__init__()
        self.name = name
        self.z_dim = 10
        self.y_dim = 10
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.y_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x, y):
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that we are interested in the ELBO of ln p(x | y)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # get encoder output
        m, v = self.enc(x, y)
        # sample gaussian using reparameterization trick
        z = ut.sample_gaussian(m, v)
        # compute p_\theta(x|z,y)
        mu_theta = self.dec(z, y)
        var_theta = torch.ones_like(mu_theta) * (1. / self.y_dim)
        # compute reconstruction loss as negative of log-likelihood: -log(p_\theta(x|z,y))
        rec = -ut.log_normal(x, mu_theta, var_theta)
        # compute KL divergence between q_\phi(z|x,y) and p(z)
        kl_z = ut.kl_normal(mu_theta, var_theta, self.z_prior[0], self.z_prior[1])

        # compute mean over all the samples
        rec = torch.mean(rec)
        kl_z = torch.mean(kl_z)
        nelbo = rec + kl_z
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, rec

    def loss(self, x, y):
        nelbo, kl_z, rec = self.negative_elbo_bound(x, y)
        loss = nelbo

        summaries = dict((
            ('train/loss', loss),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_mean_given(self, z, y):
        return self.dec(z, y)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))
