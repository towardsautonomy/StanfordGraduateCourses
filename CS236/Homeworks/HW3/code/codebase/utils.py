import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import os
import shutil
import torch
import numpy as np
import sklearn
import sklearn.datasets
from sklearn.preprocessing import StandardScaler


def save_checkpoint(state, is_best, folder="./", filename="checkpoint.pth.tar"):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename), os.path.join(folder, "model_best.pth.tar")
        )


def plot_samples(samples, data, epoch, args):
    """
    plotting code to look at both original data and model samples
    """
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    data = data.tensors[0].cpu().numpy()

    # plot real data and samples
    ax1.scatter(data[:, 0], data[:, 1], s=10)
    ax1.set_title("Real data")

    ax2.scatter(samples[:, 0], samples[:, 1], s=10)
    ax2.set_title("Generated Samples at Epoch {}".format(epoch))

    # despine then save plot
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "samples_epoch{}.png".format(epoch)))


def make_halfmoon_toy_dataset(n_samples=30000, batch_size=100):
    # lucky number
    rng = np.random.RandomState(777)

    # generate data and normalize to 0 mean
    data = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.05)[0]
    data = data.astype("float32")
    data = StandardScaler().fit_transform(data)

    # turn this into a torch dataset
    data = torch.from_numpy(data).float()

    # change this to train/val/test split
    p_idx = np.random.permutation(n_samples)
    train_idx = p_idx[0:24000]
    val_idx = p_idx[24000:27000]
    test_idx = p_idx[27000:]

    # partition data into train/valid/test
    train_dataset = torch.utils.data.TensorDataset(data[train_idx])
    val_dataset = torch.utils.data.TensorDataset(data[val_idx])
    test_dataset = torch.utils.data.TensorDataset(data[test_idx])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader
