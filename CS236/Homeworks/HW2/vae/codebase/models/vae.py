# Copyright (c) 2021 Rui Shu

import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # compute the mean and variance for latent space
        m, v = self.enc(x)
        # use reparameterization trick to sample from the latent space
        z = ut.sample_gaussian(m, v)
        # decode logits
        logits = self.dec(z)
        # compute log-likelihood
        log_prob = ut.log_bernoulli_with_logits(x=x, logits=logits)
        # reconstruction loss
        rec = -torch.mean(log_prob)
        # compute the KL divergence
        kl = torch.mean(ut.kl_normal(m, v, self.z_prior_m, self.z_prior_v))
        # compute negative ELBO
        nelbo = kl + rec

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        # compute the mean and variance for latent space
        m, v = self.enc(x)
        # duplicate the parameters `iw` times to allow `iw` sampling
        m = ut.duplicate(m, iw)
        v = ut.duplicate(v, iw)
        x = ut.duplicate(x, iw)
        # use reparameterization trick to sample from the latent space
        z = ut.sample_gaussian(m, v)
        # decode logits
        logits = self.dec(z)
        # compute the KL divergence
        kl = ut.log_normal(z, m, v) - ut.log_normal(z, self.z_prior[0], self.z_prior[1])
        # compute reconstruction loss
        rec = -ut.log_bernoulli_with_logits(x=x, logits=logits)
        # compute nelbo
        nelbo = kl + rec
        # IWAE is the mean of ELBO
        iwae = ut.log_mean_exp(-nelbo.reshape(iw, -1), dim=0)
        # compute NIWAE
        niwae = -iwae
        # compute mean
        niwae = torch.mean(niwae)
        kl = torch.mean(kl)
        rec = torch.mean(rec)

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
