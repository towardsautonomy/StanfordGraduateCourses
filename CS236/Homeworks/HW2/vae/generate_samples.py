# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
from codebase import utils as ut
from codebase.models.vae import VAE
from codebase.models.gmvae import GMVAE
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
parser.add_argument('--k',         type=int, default=500,   help="Number mixture components in MoG prior")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--model',     type=str, choices=['vae', 'gmvae'], help="Model Type", required=True)
args = parser.parse_args()
layout = [
      ('model={:s}', args.model),
      ('z={:02d}',  args.z)
  ]
if args.model == 'gmvae':
  layout.append(('k={:03d}',  args.k))
layout.append(('run={:04d}', args.run))

model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None 
if args.model == 'vae':
  model = VAE
elif args.model == 'gmvae':
  model = GMVAE

model_ = model(z_dim=args.z, name=model_name).to(device)
ut.load_model_by_name(model_, global_step=args.iter_max, device=device)

# sample from priors
n_rows = 10
n_cols = 20
z_sample = model_.sample_z(n_rows*n_cols).to(device)
digits = model_.sample_x_given(z_sample)
digits = torch.reshape(digits, (-1, 28, 28)).cpu().detach().numpy()

# plot digits
fig = plt.figure()
plt.tight_layout()
plt.suptitle('GMVAE generated samples from the set prior')
for i in range(n_rows*n_cols):
  plt.subplot(n_rows, n_cols, i+1)
  plt.imshow(digits[i], cmap='gray', interpolation='none')
  plt.xticks([])
  plt.yticks([])
plt.show()