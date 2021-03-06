import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
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
COMPUTE_EPIPOLE computes the epipole in homogenous coordinates
given matching points in two images and the fundamental matrix
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the Fundamental matrix such that (points1)^T * F * points2 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(points1, points2, F):
    # compute epipolar line
    epipolar_lines = np.dot(F.T, points2.T).T

    ## compute epipole
    # using the fact that the dot product of a line and a point on that line
    # equates to zero, we can build up a set of linear equations [l.e = 0], 
    # whre l= epipolar line and e = epipole; and then solve for e using SVD.
    u, s, v_t = np.linalg.svd(epipolar_lines, full_matrices=True)
    # epipole can be obtained as the last column of v or last row of v_transpose
    e = v_t.T[:,-1]
    # set the last entry to 1.0
    e = e / e[-1]

    return e
    
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    '''
    This function converts a vector 'a' of size 3
    to a 3x3 matrix '[a]_x' for cross product to 
    dot product conversion.
    '''
    def cross_product_mat(a):
        m = np.zeros((3,3), dtype=np.float32)
        m[1,0] = a[2]
        m[0,1] = -a[2]
        m[2,0] = -a[1]
        m[0,2] = a[1]
        m[2,1] = a[0]
        m[1,2] = -a[0]
        return m

    ## Let's first compute the homography H2 as H2 = T^(-1).GRT
    T = np.eye(3, dtype=np.float32)
    T[0,2] = -im2.shape[1] / 2.0
    T[1,2] = -im2.shape[0] / 2.0
    # translate the epipole
    Te2 = np.dot(T, e2.T)
    Te2 /= Te2[-1]
    # let's compute the rotation matrix
    alpha = 1 if Te2[0] >= 0 else 0
    R = np.eye(3, dtype=np.float32)
    R[0,0] = alpha * Te2[0]  / np.sqrt(Te2[0] ** 2 + Te2[1] ** 2)
    R[0,1] = alpha * Te2[1]  / np.sqrt(Te2[0] ** 2 + Te2[1] ** 2)
    R[1,0] = -1 * alpha * Te2[1]  / np.sqrt(Te2[0] ** 2 + Te2[1] ** 2)
    R[1,1] = alpha * Te2[0]  / np.sqrt(Te2[0] ** 2 + Te2[1] ** 2)
    RTe2 = np.dot(R, Te2)
    RTe2 /= RTe2[-1]
    # compute G
    G = np.eye(3, dtype=np.float32)
    G[-1,0] = -1. / RTe2[0]
    # compute H2
    H2 = np.dot(np.linalg.inv(T), np.dot(G, np.dot(R, T)))
    # H2 = H2 / H2[-1,-1] # up to scale
    
    ## Now, let's proceed to compute H1
    e2x = cross_product_mat(e2)
    v = np.array([1., 1., 1.], dtype=np.float32)
    e2v = np.dot(np.resize(e2, (3,1)), np.resize(v, (1,3)))
    M = np.dot(e2x, F) + e2v
    # Ha can now be computed by setting up a linear minimization problem
    p_hat = np.dot(H2, np.dot(M, points1.T)).T
    p_hat = np.array([(p_/p_[-1]) for p_ in p_hat], dtype=np.float32) # homogeneous format
    p_hat_prime = np.dot(H2, points2.T).T
    p_hat_prime = np.array([p_/p_[-1] for p_ in p_hat_prime], dtype=np.float32) # homogeneous format

    # For least-square solution of the problem [W.a = b], error is minimized by setting
    # [W_transpose.W.a = W_transpose.b] ref: https://eeweb.engineering.nyu.edu/iselesni/lecture_notes/least_squares/least_squares_SP.pdf
    # This gives us the solution: [a = {W_transpose.W}_inverse.W_transpose.b]
    # Build W and b matrices
    W = p_hat
    b = p_hat_prime[:,0]
    a = np.dot(np.dot(np.linalg.inv(np.dot(W.T, W)), W.T), b)
    # Alternatively, numpy can be used to directly compute this solution: a = np.linalg.lstsq(W, b)[0]
    Ha = np.eye(3, dtype=np.float32)
    Ha[0] = a
    # formulate H1
    H1 = np.dot(Ha, np.dot(H2, M))
    # H1 = H1 / H1[-1,-1] # up to scale
    # return homography matrices
    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'p1_data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print("H1:\n", H1)
    print('')
    print("H2:\n", H2)

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
