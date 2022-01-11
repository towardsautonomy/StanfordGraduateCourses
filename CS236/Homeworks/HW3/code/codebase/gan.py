import torch
from torch.nn import functional as F


def loss_nonsaturating_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # compute the discriminator loss
    d_loss = -torch.mean(torch.log(torch.sigmoid(d(x_real))) + \
                         torch.log(1 - torch.sigmoid(d(g(z)))))

    return d_loss

def loss_nonsaturating_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # compute the generator loss
    g_loss = -torch.mean(F.logsigmoid(d(g(z))))

    return g_loss


def conditional_loss_nonsaturating_d(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_fake = y_real  # use the real labels as the fake labels as well

    # compute the discriminator loss
    d_loss = -torch.mean(torch.log(torch.sigmoid(d(x_real, y_real))) + \
                         torch.log(1 - torch.sigmoid(d(g(z, y_fake), y_fake))))

    return d_loss


def conditional_loss_nonsaturating_g(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_fake = y_real  # use the real labels as the fake labels as well

    # compute the generator loss
    g_loss = -torch.mean(F.logsigmoid(d(g(z, y_fake), y_fake)))

    return g_loss


def loss_wasserstein_gp_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): wasserstein discriminator loss
    """
    lambda_ = 10.0
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    # generate fake data
    x_fake = g(z)
    # compute the output of discriminator
    d_fake, d_real = d(x_fake), d(x_real)
    # sample from Uniform([0, 1])
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    # compute r_theta(x)
    r_theta = alpha * x_fake + (1 - alpha) * x_real
    # pass through the discriminator
    d_r_theta = d(r_theta)

    # compute the gradient of d_r_theta
    grad_d_r_theta = torch.autograd.grad(outputs=d_r_theta, inputs=r_theta,
                                          grad_outputs=torch.ones(d_r_theta.size(), device=device),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    # compute the gradient penalty
    grad_penalty = ((grad_d_r_theta.view(grad_d_r_theta.size(0), -1).norm(dim=1) - 1) ** 2).mean()
    # compute the wasserstein loss
    d_loss = torch.mean(d_fake - d_real) + lambda_ * grad_penalty

    return d_loss


def loss_wasserstein_gp_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): wasserstein generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

      # generate fake data
    x_fake = g(z)
    # compute the output of discriminator
    d_fake = d(x_fake)
    # compute generator loss
    g_loss = -torch.mean(d_fake)

    return g_loss
