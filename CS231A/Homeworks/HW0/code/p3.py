# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def part_a():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE
    img1 = io.imread('image1.jpg')
    img2 = io.imread('image2.jpg')

    # END YOUR CODE HERE
    return img1, img2

def normalize_img(img):
    img = np.array(img, dtype='double')
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def part_b(img1, img2):
    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE
    img1 = normalize_img(img1)
    img2 = normalize_img(img2)

    # END YOUR CODE HERE
    return img1, img2
    
def part_c(img1, img2):
    # ===== Problem 3c =====
    # Add the images together and re-normalize them
    # to have minimum value 0 and maximum value 1.
    # Display this image.
    sumImage = None
    
    # BEGIN YOUR CODE HERE
    sumImage = img1 + img2
    sumImage = normalize_img(sumImage)

    # END YOUR CODE HERE
    return sumImage

def part_d(img1, img2):
    # ===== Problem 3d =====
    # Create a new image such that the left half of
    # the image is the left half of image1 and the
    # right half of the image is the right half of image2.

    newImage1 = None

    # BEGIN YOUR CODE HERE
    newImage1 = np.zeros_like(img1)
    newImage1[:,:int(newImage1.shape[1]/2)] = img1[:,:int(newImage1.shape[1]/2)]
    newImage1[:,int(newImage1.shape[1]/2):] = img2[:,int(newImage1.shape[1]/2):]

    # END YOUR CODE HERE
    return newImage1

def part_e(img1, img2):    
    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd
    # numbered row is the corresponding row from image1 and the
    # every even row is the corresponding row from image2.
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage2 = None

    # BEGIN YOUR CODE HERE
    newImage2 = img1
    for i in range(0,img1.shape[0],2):
        newImage2[i,:] = img2[i,:]

    # END YOUR CODE HERE
    return newImage2

def part_f(img1, img2):     
    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and tile may be helpful here.

    newImage3 = None

    # BEGIN YOUR CODE HERE
    newImage3 = img1
    newImage3[::2] = img2[::2]

    # END YOUR CODE HERE
    return newImage3

def part_g(img):         
    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image.
    # Display the grayscale image with a title.

    # BEGIN YOUR CODE HERE
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    img = R * 299/1000 + G * 587/1000 + B * 114/1000

    # END YOUR CODE HERE
    return img

if __name__ == '__main__':
    img1, img2 = part_a()
    img1, img2 = part_b(img1, img2)
    sumImage = part_c(img1, img2)
    newImage1 = part_d(img1, img2)
    newImage2 = part_e(img1, img2)
    newImage3 = part_f(img1, img2)
    img = part_g(newImage3)