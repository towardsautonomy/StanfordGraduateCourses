# CS231A Homework 0, Problem 4
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def part_a():
    # ===== Problem 4a =====
    # Read in image1 as a grayscale image. Take the singular value
    # decomposition of the image.

    img1 = None

    # BEGIN YOUR CODE HERE
    img1 = np.array(io.imread('image1.jpg', as_gray=True))
    u, s, v = np.linalg.svd(img1, full_matrices=True)

    # END YOUR CODE HERE
    return u, s, v

def part_b(u, s, v):
    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.

    rank1approx = None

    # BEGIN YOUR CODE HERE
    k = 1
    # extract top k vectors in singular matrices and top k values in singular values
    uk = u[:,:k]
    sk = s[:k]
    vk = v[:k,:]
    # perform low-rank approximation
    sk_tiled = np.transpose(np.tile(sk, (vk.shape[1], 1)))
    sv = np.multiply(sk_tiled, vk)
    rank1approx = np.dot(uk, sv)

    # END YOUR CODE HERE
    return rank1approx

def part_c(u, s, v):
    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.

    rank20approx = None

    # BEGIN YOUR CODE HERE
    k = 20
    # extract top k vectors in singular matrices and top k values in singular values
    uk = u[:,:k]
    sk = s[:k]
    vk = v[:k,:]
    # perform low-rank approximation
    sk_tiled = np.transpose(np.tile(sk, (vk.shape[1], 1)))
    sv = np.multiply(sk_tiled, vk)
    rank20approx = np.dot(uk, sv)

    # END YOUR CODE HERE
    return rank20approx

if __name__ == '__main__':
    u, s, v = part_a()
    rank1approx = part_b(u, s, v)
    rank20approx = part_c(u, s, v)