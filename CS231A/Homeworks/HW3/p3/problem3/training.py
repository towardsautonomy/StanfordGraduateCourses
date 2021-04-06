import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def train(dataset, model, batch_size, epochs):
    """
    A function that trains the given model on the provided dataset.
    """
    # initialize a DataLoader on the dataset with the appropriate batch
    # size and shuffling enabled.
    data_loader = DataLoader(dataset, batch_size=batch_size)

    # initialize this to be a Cross Entropy Classification loss.
    criterion = nn.CrossEntropyLoss()

    # initialize this to be an Stochastic gradient descent optimizer with
    # learning rate set to 0.001 and momentum set to 0.9.
    optimizer = optim.SGD(model.parameters, lr=0.001, momentum=0.9)

    losses = []
    accuracies = []
    for e in range(epochs):
        running_loss = 0
        correct_count = 0
        for images, labels in data_loader:
            # Training pass
            optimizer.zero_grad()
            images_cuda = images.cuda()
            labels_cuda = labels.cuda()

            # call the model's classify method with images_cuda.
            output = model.classify(images_cuda)

            # call the criterion with the output and labels_cuda.
            loss = criterion(output, labels_cuda)

            # Calculate accuracy
            _, predictions = torch.max(output, 1)

            # calculate the number of correctly classified inputs.
            num_correct = sum(predictions == labels_cuda)

            correct_count += num_correct

            if e > 0: # Don't optimize at epoch 0, to see how model starts
                # use the loss to back propagate errors, and then do an
                # optimization step.
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        # calculate the average loss in this epoch.
        epoch_loss = running_loss / len(data_loader)

        # calculate the float accuracy of this epoch.
        accuracy = correct_count / len(dataset)

        losses.append(epoch_loss)
        accuracies.append(accuracy)
        print("Epoch %d - Training loss: %.3f , Training Accuracy: %.3f" %
              (e, epoch_loss, accuracy))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses)
    ax2.plot(accuracies)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    plt.show()
