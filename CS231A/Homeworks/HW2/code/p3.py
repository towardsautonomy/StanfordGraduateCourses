import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    # let's find centroid of points in their respective cameras.
    points_im1_centroid = np.array((np.mean(points_im1[:,0]), np.mean(points_im1[:,1])), dtype=np.float32) 
    points_im2_centroid = np.array((np.mean(points_im2[:,0]), np.mean(points_im2[:,1])), dtype=np.float32)
    
    # compute centered points
    x1j_hat = points_im1[:,:2] - points_im1_centroid
    x2j_hat = points_im2[:,:2] - points_im2_centroid

    # build the measurement matrix
    D = []
    # camera 1
    D.append([x1j_hat[j,0] for j in range(x1j_hat.shape[0])])
    D.append([x1j_hat[j,1] for j in range(x1j_hat.shape[0])])
    # camera 2
    D.append([x2j_hat[j,0] for j in range(x2j_hat.shape[0])])
    D.append([x2j_hat[j,1] for j in range(x2j_hat.shape[0])])
    D = np.array(D, dtype=np.float32)
    
    # This D can be factorized as [D=MS]; where M=Motion, S=Structure
    # let's decompose D using SVD
    u, s, v_t = np.linalg.svd(D, full_matrices=True)
    # For D=MS, M captures camera parameters (2m x 3), and S captures 3D points ( 3 x n)
    motion = u[:,:3]
    s_mat = np.diag(s[:3])
    structure = np.dot(s_mat, v_t[:3,:])

    # return motion and structure
    return structure, motion

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        plt.show()
