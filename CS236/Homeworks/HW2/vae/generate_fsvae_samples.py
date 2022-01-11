# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
from codebase import utils as ut
from codebase.models.fsvae import FSVAE
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--iter_max',  type=int, default=1000000, help="Number of training iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
args = parser.parse_args()
layout = [
      ('model={:s}', 'fsvae'),
      ('run={:04d}', args.run)
  ]

model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_ = FSVAE(name=model_name).to(device)
ut.load_model_by_name(model_, global_step=args.iter_max, device=device)

# plot digits
fig = plt.figure()
plt.tight_layout()
plt.suptitle('FSVAE generated samples for digits with various styles')

# sample from priors
n_z_samples = 20
z_sample = model_.sample_z(n_z_samples).to(device)
for digit_i in range(model_.y_dim):
  y_labels = torch.Tensor(np.repeat(digit_i, n_z_samples)).int()
  y_labels = y_labels.new(np.eye(10)[y_labels]).to(device).float()
  mu_theta = model_.compute_mean_given(z_sample, y_labels)
  # clip to within [0, 1] range
  mu_theta = torch.clip(mu_theta, 0., 1.)
  digits = torch.reshape(mu_theta, (-1, 3, 32, 32)).cpu().detach().numpy()
  digits = np.transpose(digits, (0,2,3,1))

  # plot digits
  for j in range(digits.shape[0]):
    plt.subplot(model_.y_dim, n_z_samples, digit_i*digits.shape[0]+j+1)
    plt.imshow(digits[j], interpolation='none')
    plt.xticks([])
    plt.yticks([])

plt.show()