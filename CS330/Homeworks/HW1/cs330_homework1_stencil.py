# -*- coding: utf-8 -*-
"""CS330_Homework1_Stencil.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bsjJzQCAm9Vcy2L0DZl5oK5PiWmxrfX2

##Setup

You will need to make a copy of this Colab notebook in your Google Drive before you can edit the homework files. You can do so with **File &rarr; Save a copy in Drive**.
"""

import os
from google_drive_downloader import GoogleDriveDownloader as gdd

# Need to download the Omniglot dataset -- DON'T MODIFY THIS CELL
if not os.path.isdir('./omniglot_resized'):
    gdd.download_file_from_google_drive(file_id='1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
                                        dest_path='./omniglot_resized.zip',
                                        unzip=True)
    
assert os.path.isdir('./omniglot_resized')

import numpy as np
import os
import random
import glob
import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt
from collections import deque
import csv

def get_images(paths, labels, nb_samples=None, shuffle=False):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]

    if shuffle:
        random.shuffle(images_labels)
    return images_labels

def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    import imageio
    image = imageio.imread(filename)  # misc.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image

def convert_to_one_hot(label, num_classes):
    """
    Takes a label integer and returns one-hot vector
    Args:
        label: integar specifying class 
        num_classes: number of classes
    """
    onehot_ = np.zeros(num_classes, dtype=np.float32)
    onehot_[label] = 1
    return onehot_

class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}):
        """
        Args:
            num_classes: Number of classes for classification (K-way)
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes
        self.shuffle_only_end_task = True

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]
        character_folders = sorted(character_folders)
 
        random.seed(1)
        random.shuffle(character_folders)

        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]

    def sample_batch(self, batch_type, batch_size, shuffle=True):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        # Sample characters
        batch_classes_folders = np.asarray(random.sample(folders, self.num_classes))
        batch_classes_labels = range(0, self.num_classes)

        all_image_batches = np.zeros((batch_size, self.num_samples_per_class, self.num_classes, self.dim_input))
        all_label_batches = np.zeros((batch_size, self.num_samples_per_class, self.num_classes, self.dim_output))

        for batch in range(batch_size):
          images_labels_ = np.array(get_images(batch_classes_folders, batch_classes_labels, nb_samples=self.num_samples_per_class))
          images_labels_reshaped = np.reshape(images_labels_, newshape=(self.num_classes, self.num_samples_per_class, 2))

          # organize labels
          images_labels_reshaped = np.array([images_labels_reshaped[i,j] 
                                              for j in range(self.num_samples_per_class)
                                              for i in range(self.num_classes)])
          images_labels_reshaped = np.reshape(images_labels_reshaped, newshape=(self.num_samples_per_class, self.num_classes, 2))
          # extract image filenames and labels
          img_fnames_ = np.array([images_labels_reshaped[i,j,1] 
                                  for i in range(self.num_samples_per_class)
                                  for j in range(self.num_classes)])
          labels_ = np.array([images_labels_reshaped[i,j,0] 
                                for i in range(self.num_samples_per_class)
                                for j in range(self.num_classes)])

          # build batch of images
          imgs_batch = np.array([image_file_to_array(img_fnames_[i], self.dim_input) 
                                  for i in range(len(img_fnames_))])

          all_image_batches[batch] = np.reshape(imgs_batch, newshape=(self.num_samples_per_class, self.num_classes, self.dim_input))
          
          # build batch of labels
          onehot_batch = np.array([convert_to_one_hot(int(labels_[i]), self.num_classes)
                                    for i in range(len(labels_))])
          all_label_batches[batch] = np.reshape(onehot_batch, newshape=(self.num_samples_per_class, self.num_classes, self.dim_output))

          # shuffle
          if shuffle == True:
            if self.shuffle_only_end_task == True:
              indices = np.arange(self.num_classes)
              np.random.shuffle(indices)
              all_image_batches[batch][-1,:,:] = all_image_batches[batch][-1,indices,:]
              all_label_batches[batch][-1,:,:] = all_label_batches[batch][-1,indices,:]
            else:
              # shuffle the images and labels within each sample 'K'
              for i in range(self.num_samples_per_class):
                indices = np.arange(self.num_classes)
                np.random.shuffle(indices)
                all_image_batches[batch][i,:,:] = all_image_batches[batch][i,indices,:]
                all_label_batches[batch][i,:,:] = all_label_batches[batch][i,indices,:]

        # return a batch of flattened images and corresponding labels
        return all_image_batches.astype(np.float32), all_label_batches.astype(np.float32)

"""**Test the Data Generator**"""

config = {'data_folder':'./omniglot_resized', 'img_size':(28, 28)}
num_classes=5
num_samples_per_class=3
batch_size=4
dloader = DataGenerator(num_classes=num_classes, num_samples_per_class=num_samples_per_class, config=config)
im, l = dloader.sample_batch(batch_type='train', batch_size=batch_size)

# test the method of preparing labels to be fed to the network
input_labels_gt = tf.identity(l[:,:-1,:,:])
# concatenate zeros for last label
l = tf.concat([input_labels_gt, tf.zeros((l.shape[0], 1, num_classes, num_classes), tf.float32)], axis=1).numpy()

# plot few samples for verification
plt.figure(figsize=(16,12))
for idx_s in range(num_samples_per_class):
  for idx_c in range(num_classes):
    plt.subplot(num_samples_per_class,num_classes,idx_s*num_classes+idx_c+1)
    img = np.reshape(im[0,idx_s,idx_c,:], newshape=(28,28))
    plt.imshow(img, cmap='gray')
    plt.title('class={}'.format(np.argmax(l[0,idx_s,idx_c,:])))
    if idx_s == num_samples_per_class-1:
        plt.ylabel('test')
    else:
        plt.ylabel('train')
plt.show()

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers

class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.reshape = tf.keras.layers.Reshape((int(num_classes*samples_per_class),-1))
        self.reshape_back = tf.keras.layers.Reshape((samples_per_class,num_classes,-1))
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)
        self.ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    @tf.function
    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        # set label for last sample in each class to zero
        input_labels_gt = tf.identity(input_labels[:,:-1,:,:])
        # concatenate zeros for last label
        input_labels_modified = tf.concat([input_labels_gt, tf.zeros((input_labels.shape[0], 1, self.num_classes, self.num_classes), tf.float32)], axis=1)
        # reshape images and labels to build a 3-dimensional vector for LSTM
        input_labels_modified = self.reshape(input_labels_modified)
        input_images = self.reshape(input_images)
        # concatenate images with labels
        concatenated_input = tf.concat([input_images, input_labels_modified], axis=-1)
        out = self.layer1(concatenated_input)
        out = self.layer2(out)
        # #############################
        return out

    @tf.function
    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        """
        # extract last sample within each character and compute loss
        labels = self.reshape_back(labels)
        preds = self.reshape_back(preds)
        loss_ = self.ce(labels, preds)
        return loss_

        # loss_ = tf.zeros(1, tf.float32)
        # for i in range(self.num_classes):
        #     label_end_sample_gt = labels[:,-1,i,:]
        #     label_end_sample_pred = preds[:,-1,i,:]
        #     loss_ += self.ce(label_end_sample_gt, label_end_sample_pred)
        # return loss_ / tf.constant(self.num_classes, dtype=tf.float32)

@tf.function
def cross_entropy(x, y, epsilon = 1e-9):
    return -tf.reduce_mean((y * tf.math.log(x + epsilon)) + ((1.0 - y) * tf.math.log((1.0 - x) + epsilon)))

@tf.function
def mse(x, y):
    return tf.reduce_mean((x - y) * (x - y))

@tf.function
def train_step(images, labels, model, optim, eval=False):
    with tf.GradientTape() as tape:
        predictions = model(images, labels)
        loss = model.loss_function(predictions, labels)
    if not eval:
        gradients = tape.gradient(loss, model.trainable_variables)
        optim.apply_gradients(zip(gradients, model.trainable_variables))
    return predictions, loss


def main(num_classes=5, num_samples=1, meta_batch_size=16, random_seed=1234):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    data_generator = DataGenerator(num_classes, num_samples + 1)

    o = MANN(num_classes, num_samples + 1)
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)

    steps = []
    losses = []
    accuracies = []
    for step in range(200):
        i, l = data_generator.sample_batch('train', meta_batch_size)

        _, ls = train_step(i, l, o, optim)

        if (step + 1) % 100 == 0:
            print("*" * 5 + "Iter " + str(step + 1) + "*" * 5)
            i, l = data_generator.sample_batch('test', 100)
            pred, tls = train_step(i, l, o, optim, eval=True)
            print("Train Loss:", ls.numpy(), "Test Loss:", tls.numpy())
            pred = tf.reshape(pred, [-1, num_samples + 1, num_classes, num_classes])
            pred = tf.math.argmax(pred[:, -1, :, :], axis=2)
            l = tf.math.argmax(l[:, -1, :, :], axis=2)
            acc = tf.reduce_mean(tf.cast(tf.math.equal(pred, l), tf.float32)).numpy()
            print("Test Accuracy", acc)

            steps.append(step)
            losses.append([ls.numpy(), tls.numpy()])
            accuracies.append(acc)

    return np.asarray(steps), np.asarray(losses), np.asarray(accuracies)

# compute moving average
def moving_average(x, n):
    # create a deque object
    x_deque = deque(maxlen=n)
    ma_x = []
    for x_ in x:
        x_deque.append(x_)
        ma_x.append(np.mean(x_deque))
    return np.array(ma_x)

if __name__ == '__main__':
    num_samples_per_class=1
    num_classes=3
    meta_batch_size=16
    logdir = 'logs'
    plotdir = 'plots'

    # filenames
    csv_fname = os.path.join(logdir, 'log_{}way_{}shot.csv'.format(num_classes, num_samples_per_class))
    plot_fname = os.path.join(plotdir, 'plot_{}way_{}shot.png'.format(num_classes, num_samples_per_class))
    # create folders if it does not exist
    if not os.path.exists(logdir):
        os.system('mkdir -p {}'.format(logdir))
    if not os.path.exists(plotdir):
        os.system('mkdir -p {}'.format(plotdir))

    with open(csv_fname, 'a+', newline='') as csvfile:
        fieldnames = ['iter',       \
                    'train_loss', \
                    'test_loss',  \
                    'acc'         ]
        # write header
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()

        steps, loss, acc = main(num_classes=num_classes, num_samples=num_samples_per_class, meta_batch_size=meta_batch_size, random_seed=1234)

        for i in range(len(steps)):
            if not os.path.exists(csv_fname):
                csv_writer.writerow({   'iter'          : steps[i],   \
                                        'train_loss'    : loss[i,0],  \
                                        'test_loss'     : loss[i,1],  \
                                        'acc'           : acc[i]        })

        moving_avg_test_loss = moving_average(loss[:,1], 2)
        moving_avg_test_acc = moving_average(acc, 2)
        print('Achieved test accuracy: {:.2f}%'.format(np.max(acc[-5:])*100.0))

        # display training results
        plt.figure(figsize=(16,9))
        plt.title('Loss and Accuracy [K={}, N={}]'.format(num_samples_per_class, num_classes))
        plt.subplot(2,1,1)
        plt.plot(steps, loss[:,0])
        plt.plot(steps, loss[:,1])
        plt.plot(steps, moving_avg_test_loss[:len(steps)])
        plt.grid()
        plt.legend(['Train Loss', 'Test Loss', 'Avg Test Loss'])
        plt.xlabel('iterations')

        plt.subplot(2,1,2)
        plt.plot(steps, acc)
        plt.plot(steps, moving_avg_test_acc[:len(steps)])
        plt.grid()
        plt.legend(['Test Accuracy', 'Avg Test Accuracy'])
        plt.xlabel('iteration')

        plt.savefig(plot_fname)
        plt.show()