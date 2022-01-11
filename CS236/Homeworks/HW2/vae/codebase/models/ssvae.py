# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
import torch.utils.data
from codebase import utils as ut
from codebase.models import nns
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class SSVAE(nn.Module):
    def __init__(self, nn='v1', name='ssvae', gen_weight=1, class_weight=100):
        super().__init__()
        self.name = name
        self.z_dim = 64
        self.y_dim = 10
        self.gen_weight = gen_weight
        self.class_weight = class_weight
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.y_dim)
        self.cls = nn.Classifier(self.y_dim)

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
        # Compute negative Evidence Lower Bound and its KL_Z, KL_Y and Rec decomposition
        #
        # To assist you in the vectorization of the summation over y, we have
        # the computation of q(y | x) and some tensor tiling code for you.
        #
        # Note that nelbo = kl_z + kl_y + rec
        #
        # Outputs should all be scalar
        ################################################################################
        y_logits = self.cls(x)
        y_logprob = F.log_softmax(y_logits, dim=1) # log(q_\phi(y|x))
        y_prob = torch.softmax(y_logprob, dim=1) # q_\phi(y|x)
        # Duplicate y based on x's batch size. Then duplicate x
        # This enumerates all possible combination of x with labels (0, 1, ..., 9)
        y = np.repeat(np.arange(self.y_dim), x.size(0))
        y = x.new(np.eye(self.y_dim)[y])
        x = ut.duplicate(x, self.y_dim)

        # compute q_\phi(z|x,y)
        m, v = self.enc(x, y)
        z = ut.sample_gaussian(m, v)
        # compute logits
        logits = self.dec(z, y)
        # compute reconstruction loss: -log p_\theta(x|z, y)
        rec_i = -ut.log_bernoulli_with_logits(x=x, logits=logits)
        # compute expectation of above under q_\phi(y|x) gives us the KL divergence
        rec = torch.sum(y_prob * torch.transpose(torch.reshape(rec_i, (self.y_dim, -1)), 0, 1), dim=-1)

        # compute categorical KL divergence: kl = q.log(q/p); p is the log of samples 
        # from uniform distribution over all categories, i.e. p(y) = Categorical(y) = 1/10
        kl_y = ut.kl_cat(y_prob, y_logprob, np.log(1.0 / self.y_dim))

        # compute KL divergence between q_\phi(z|x,y) and p(z)
        kl_z_i = ut.kl_normal(m, v, self.z_prior[0], self.z_prior[1])
        # expectation of above under q_\phi(y|x) gives us the KL divergence
        kl_z = torch.sum(y_prob * torch.transpose(torch.reshape(kl_z_i, (self.y_dim, -1)), 0, 1), dim=-1)

        # compute mean over all samples in the batch
        kl_y = torch.mean(kl_y)
        kl_z = torch.mean(kl_z)
        rec = torch.mean(rec)

        # compute nelbo
        nelbo = kl_y + kl_z + rec

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, kl_y, rec

    def classification_cross_entropy(self, x, y):
        y_logits = self.cls(x)
        return F.cross_entropy(y_logits, y.argmax(1))

    def loss(self, x, xl, yl):
        if self.gen_weight > 0:
            nelbo, kl_z, kl_y, rec = self.negative_elbo_bound(x)
        else:
            nelbo, kl_z, kl_y, rec = [0] * 4
        ce = self.classification_cross_entropy(xl, yl)
        loss = self.gen_weight * nelbo + self.class_weight * ce

        summaries = dict((
            ('train/loss', loss),
            ('class/ce', ce),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/kl_y', kl_y),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_sigmoid_given(self, z, y):
        logits = self.dec(z, y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

    def sample_x_given(self, z, y):
        return torch.bernoulli(self.compute_sigmoid_given(z, y))
