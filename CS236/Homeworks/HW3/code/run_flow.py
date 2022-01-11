import argparse
import os

import tqdm
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from codebase.flow_network import MAF
from codebase.utils import make_halfmoon_toy_dataset, save_checkpoint, plot_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a MAF model.")
    parser.add_argument("--device", default="cpu", help="[cpu,cuda]")
    parser.add_argument(
        "--n_flows", default=5, type=int, help="number of planar flow layers"
    )
    parser.add_argument(
        "--hidden_size",
        default=100,
        type=int,
        help="number of hidden units in each flow layer",
    )
    parser.add_argument(
        "--n_epochs", default=50, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--n_samples",
        default=30000,
        type=int,
        help="total number of data points in toy dataset",
    )
    parser.add_argument("--batch_size", default=100, type=int, help="batch size")
    parser.add_argument("--out_dir", default="maf", help="path to output directory")
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(777)
    np.random.seed(777)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    # load half-moons dataset
    train_loader, val_loader, test_loader = make_halfmoon_toy_dataset(
        args.n_samples, args.batch_size
    )

    # load model
    model = MAF(
        input_size=2, hidden_size=args.hidden_size, n_hidden=1, n_flows=args.n_flows
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    def train(epoch):
        model.train()
        total_loss = 0.0
        batch_idx = 0.0

        for data in tqdm.tqdm(train_loader):
            batch_idx += 1
            if isinstance(data, list):
                data = data[0]
            batch_size = len(data)
            data = data.view(batch_size, -1)
            data = data.to(device)

            # run MAF
            loss = model.loss(data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save stuff
            total_loss += loss.item()

        total_loss /= batch_idx + 1
        print("Average train log-likelihood: {:.6f}".format(-total_loss))

        return total_loss

    def test(epoch, split, loader):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                if isinstance(data, list):
                    data = data[0]
                batch_size = len(data)
                data = data.view(batch_size, -1)
                data = data.to(device)

                # run through model
                loss = model.loss(data)

                # save values
                total_loss += loss.item()

            total_loss /= batch_idx + 1
            print("Average {} log-likelihood: {:.6f}".format(split, -total_loss))

        return total_loss

    # logger
    best_loss = np.inf
    train_loss_db = np.zeros(args.n_epochs + 1)
    val_loss_db = np.zeros(args.n_epochs + 1)

    # snippet of real data for plotting
    data_samples = test_loader.dataset

    for epoch in range(1, args.n_epochs + 1):
        print("Epoch {}:".format(epoch))
        train_loss = train(epoch)
        val_loss = test(epoch, "validation", val_loader)

        train_loss_db[epoch] = train_loss
        val_loss_db[epoch] = val_loss

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if is_best:
            best_epoch = epoch
        print(
            "Best validation log-likelihood at epoch {}: {:.6f}".format(
                best_epoch, -best_loss
            )
        )

        if epoch % 10 == 0:
            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cmd_line_args": args,
                },
                is_best,
                folder=args.out_dir,
            )
            # save samples
            samples = model.sample(device, n=1000)
            plot_samples(samples, data_samples, epoch, args)
    test_loss = test(epoch, "test", test_loader)

    # save stuff
    np.save(os.path.join(args.out_dir, "train_loss.npy"), train_loss_db)
    np.save(os.path.join(args.out_dir, "val_loss.npy"), val_loss_db)
    np.save(os.path.join(args.out_dir, "test_loss.npy"), test_loss)
