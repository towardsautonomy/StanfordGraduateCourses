import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
'''
def lls_eight_point_alg(points1, points2):
    # points1 and points2 are in homogeneous coordinate system.
    # let's build the set of linear equations [Wf = 0]
    # where W is composed of W = [uu', uv', u, vu', vv', v, u', v', 1]
    # For a set of points p and p', fundamental matrix relates
    # them as [pT.F.p' = 0]
    # Here, p = points2; p' = points1
    W = [] 
    for i, p in enumerate(points2):
        p_prime = points1[i]
        W.append([p[0]*p_prime[0],p[0]*p_prime[1],p[0],p[1]*p_prime[0],p[1]*p_prime[1],p[1],p_prime[0],p_prime[1],p_prime[2]])
    W = np.array(W, dtype=np.float64)
    u, s, v_t = np.linalg.svd(W, full_matrices=True)
    # fundamental matrix can be obtained as the last column of v or last row of v_transpose
    F_hat = v_t.T[:,-1]
    F_hat = np.reshape(F_hat, (3,3))
    # this fundamental matrix may be full rank i.e rank=3. but the rank of fundamental matrix should be rank(f)=2
    # let's use SVD on F_hat again and then obtain a rank2 fundamental matrix
    u, s, v_t = np.linalg.svd(F_hat, full_matrices=True)
    # let's build a matrix from the first two singular values
    s_mat = np.diag(s)
    s_mat[-1,-1] = 0.
    # let's compose our rank(2) fundamental matrix
    F = np.dot(u, np.dot(s_mat, v_t))
    # return the fundamental matrix

    return F

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
'''
def normalized_eight_point_alg(points1, points2):
    # compute the centroid of all points in camera1 and camera2
    centroid1 = np.array([np.mean(points1[:,0]), np.mean(points1[:,1]), np.mean(points1[:,2])])
    centroid2 = np.array([np.mean(points2[:,0]), np.mean(points2[:,1]), np.mean(points2[:,2])])
    # let's compute mean-squared distance between each point
    # and centroid of all points in respective cameras
    # camera 1
    squared_dist1 = []
    for pt in points1:
        squared_dist1.append((pt[0] - centroid1[0]) ** 2 + \
                             (pt[1] - centroid1[1]) ** 2 + \
                             (pt[2] - centroid1[2]) ** 2)
    mean_squared_dist1 = np.sqrt(np.mean(squared_dist1))

    # camera 2
    squared_dist2 = []
    for pt in points2:
        squared_dist2.append((pt[0] - centroid2[0]) ** 2 + \
                             (pt[1] - centroid2[1]) ** 2 + \
                             (pt[2] - centroid2[2]) ** 2)
    mean_squared_dist2 = np.sqrt(np.mean(squared_dist2))

    # build a transformation matrix which will first translate these
    # points to the centroid and then scale them so that the
    # points are centered at the centroid with a mean-squared distance
    # of 2 pixels
    translation1 = np.array([[1., 0., -centroid1[0]],
                             [0., 1., -centroid1[1]],
                             [0., 0., 1.]], dtype=np.float64)
    scaling1 = np.array([[np.sqrt(2.)/mean_squared_dist1, 0., 0.],
                         [0., np.sqrt(2.)/mean_squared_dist1, 0.],
                         [0., 0., 1.0]], dtype=np.float64)
    T1 = np.dot(scaling1, translation1)
    normalized_points1 = np.dot(T1, points1.T).T

    # normalize points in second camera
    translation2 = np.array([[1., 0., -centroid2[0]],
                             [0., 1., -centroid2[1]],
                             [0., 0., 1.]], dtype=np.float64)
    scaling2 = np.array([[np.sqrt(2.)/mean_squared_dist2, 0., 0.],
                         [0., np.sqrt(2.)/mean_squared_dist2, 0.],
                         [0., 0., 1.0]], dtype=np.float64)
    T2 = np.dot(scaling2, translation2)
    normalized_points2 = np.dot(T2, points2.T).T

    # compute fundamental matrix for normalized points
    F_normalized = lls_eight_point_alg(normalized_points1, normalized_points2)

    # denormalize fundamenta matrix
    F = np.dot(T2.T, np.dot(F_normalized, T1))

    # return fundamental matrix
    return F

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):

    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.T.dot(points2.T)
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a,b,c = line
            xs = [1, im.shape[1]-1]
            ys = [(-c-a*x)/b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x,y,_ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

    # We change the figsize because matplotlib has weird behavior when 
    # plotting images of different sizes next to each other. This
    # fix should be changed to something more robust.
    new_figsize = (8 * (float(max(im1.shape[1], im2.shape[1])) / min(im1.shape[1], im2.shape[1]))**2 , 6)
    fig = plt.figure(figsize=new_figsize)
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')

'''
COMPUTE_DISTANCE_BETWEEN_POINTS_AND_EPIPOLAR_LINES
computes distance between points in each camera to the epipolar
lines computed from the other camera using fundamental matrix.
Arguments:
    points - N points in the image
    lines - lines computed from the other camera using fundamental matrix.

Returns: 
    distance for each point and corresponding line
'''
def point2line_dist(points, lines):
    distances = []
    for i, pt in enumerate(points):
        dist = abs(lines[i][0]*pt[0] + lines[i][1]*pt[1] + lines[i][2]) / np.sqrt((lines[i][0] ** 2) + (lines[i][1] ** 2))
        distances.append(dist)
    distances = np.array(distances, dtype=np.float64)
    return distances

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines. Compute just the average distance
from points1 to their corresponding epipolar lines (which you get from points2).
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # compute epipolar line
    epipolar_lines1 = np.dot(F.T, points2.T).T
    distances1 = point2line_dist(points1, epipolar_lines1)
    average_distance = np.mean(distances1)
    # return average distance
    return average_distance
'''
This function computes the distance between points and their
corresponding epipolar lines in both images.
'''
# def compute_distance_to_epipolar_lines(points1, points2, F):
#     # compute epipolar line
#     epipolar_lines1 = np.dot(F.T, points2.T).T
#     epipolar_lines2 = np.dot(F, points1.T).T
#     distances1 = point2line_dist(points1, epipolar_lines1)
#     distances2 = point2line_dist(points2, epipolar_lines2)
#     average_distance = np.mean(np.concatenate([distances1, distances2]))
#     # return average distance
#     return average_distance

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('-'*80)
        print("Set:", im_set)
        print('-'*80)

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized)
        print("Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
