import matplotlib.pyplot as plt
import random
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import random

class MNISTDataset(Dataset):
    """
    A Dataset for learning with subsets of the MNIST dataset for either the
    original labels or labels that describe how the image has been rotated.
    Rotations will be applied clockwise, with a random choice of one of the
    following degrees: [0, 45, 90, 135, 180, 225, 270, 315]

    - file - MNIST processed .pt file.
    - pct - percent of data to use
    - classify_digit_type - False=Use rotation labels.
                            True=Use original classification labels.
    """

    def __init__(self, file, pct, classify_digit_type):
        data = torch.load(file)
        self.imgs = data[0]
        self.labels = data[1]
        self.pct = pct
        self.classify_digit_type = classify_digit_type
        self.tensorToImage = torchvision.transforms.ToPILImage()
        self.imageToTensor = torchvision.transforms.ToTensor()
        self.rot_choices = [0, 45, 90, 135, 180, 225, 270, 315]
        # number of images to use
        n_images = int(self.__len__() * min(self.pct, 1.0))
        # shuffle and select a subset of dataset
        p = np.random.permutation(self.__len__())
        self.imgs = (self.imgs[p])[:n_images]
        self.labels = (self.labels[p])[:n_images]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Returns a 32x32 MNIST digit and its corresponding digit
        label, or if self.classify_digit_type is true returns
        the digit rotated by a random rotation amount from
        self.rot_choices

        for the latter, wrap the rotation integer by
        torch.tensor(rotation_integer).long()

        for resizing the image to 32x32, use tensorToImage
        so you could use PIL's resize function with resample=1
        """
        img = self.imgs[idx].unsqueeze(0)
        img = self.tensorToImage(img)
        img = img.resize((32, 32), resample=1)
        img = self.imageToTensor(img)
        label = None

        if not self.classify_digit_type:
            # 8 classes for rotation

            img = self.tensorToImage(img)
            angle_idx = random.choice(range(len(self.rot_choices)))
            # Rotate the image with a random choice in self.rot_choices
            # and assign the label accordingly.
            img = torchvision.transforms.functional.rotate(img, self.rot_choices[angle_idx])
            img = self.imageToTensor(img)
            label = torch.tensor(angle_idx).long()

        else:
            label = self.labels[idx]

        return img, label

    def show_batch(self, n=3):
        fig, axs = plt.subplots(n, n)
        fig.tight_layout()
        for i in range(n):
            for j in range(n):
                rand_idx = random.randint(0, len(self)-1)
                img, label = self.__getitem__(rand_idx)
                axs[i, j].imshow(self.tensorToImage(img), cmap='gray')
                if self.classify_digit_type:
                    axs[i, j].set_title(
                        'Label: {0} (Digit #{1})'.format(
                            label.item(), label.item()))
                else:
                    axs[i, j].set_title(
                        'Label: {0} ({1} Degrees)'.format(
                            label.item(), label.item()*45))
                axs[i, j].axis('off')
