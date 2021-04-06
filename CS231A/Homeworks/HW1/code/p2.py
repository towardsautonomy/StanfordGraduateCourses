# CS231A Homework 1, Problem 2
import numpy as np
# import cv2

'''
DATA FORMAT

In this problem, we provide and load the data for you. Recall that in the original
problem statement, there exists a grid of black squares on a white background. We
know how these black squares are setup, and thus can determine the locations of
specific points on the grid (namely the corners). We also have images taken of the
grid at a front image (where Z = 0) and a back image (where Z = 150). The data we
load for you consists of three parts: real_XY, front_image, and back_image. For a
corner (0,0), we may see it at the (137, 44) pixel in the front image and the
(148, 22) pixel in the back image. Thus, one row of real_XY will contain the numpy
array [0, 0], corresponding to the real XY location (0, 0). The matching row in
front_image will contain [137, 44] and the matching row in back_image will contain
[148, 22]
'''

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
def compute_camera_matrix(real_XY, front_image, back_image):
    # compose real homogeneous coordinates
    real_XYZ_front = np.zeros((real_XY.shape[0], 4))
    real_XYZ_front[:,0:2] = real_XY
    real_XYZ_front[:,2] = 0.0 # z = 0
    real_XYZ_front[:,3] = 1.0 # addition of 4th dimension in homogeneous coordinate system

    real_XYZ_back = np.zeros((real_XY.shape[0], 4))
    real_XYZ_back[:,0:2] = real_XY
    real_XYZ_back[:,2] = 150.0 # z = 150
    real_XYZ_back[:,3] = 1.0 # addition of 4th dimension in homogeneous coordinate system

    # put together the real coordinates
    real_XYZ_homogeneous = np.vstack((real_XYZ_front, real_XYZ_back))

    # put together the corresponding image projection coordinates
    projected_XY = np.vstack((front_image, back_image))

    # This gives us [projected_XY_homogeneous = M.real_XYZ_homogeneous]
    # M is a 3x4 unknown matrix. With the Affine assumption, last row of M = [0, 0, 0, 1]
    # This leaves us 8 unknown elements to compute.
    # Because of the Affine assumption, last row of M = [0, 0, 0, 1], and hence, 
    # projected_XY_homogeneous[2] = 1.0. This means [projected_XY_euclidean = projected_XY_homogeneous[0:2]]
    # We need to rearrange to a form [P.m = p] where 'm' is a 8x1 unknown vector to be determined, 
    # and P is [2*n x 8] known elements, where n is number of corresponding points available. 
    P = np.zeros((projected_XY.shape[0]*2, 8))
    p = np.zeros((projected_XY.shape[0]*2))
    for n in range(projected_XY.shape[0]):
        # (2*n)th row
        P[2*n, 0:4] = real_XYZ_homogeneous[n]
        p[2*n] = projected_XY[n, 0]
        # (2*n+1)th row
        P[2*n+1, 4:8] = real_XYZ_homogeneous[n]
        p[2*n+1] = projected_XY[n, 1]

    # For least-square solution of the problem [P.m = p], error is minimized by setting
    # [P_transpose.P.m = P_transpose.p] ref: https://eeweb.engineering.nyu.edu/iselesni/lecture_notes/least_squares/least_squares_SP.pdf
    # This gives us the solution: [m = {P_transpose.P}_inverse.P_transpose.p]
    # Compute the least-square solution for [P.m = p]
    P_transpose = np.transpose(P)
    m = np.dot(np.dot(np.linalg.inv(np.dot(P_transpose, P)), P_transpose), p)

    # Alternatively, numpy can be used to directly compute this solution: m = np.linalg.lstsq(P, p)[0]

    # Affine Camera Matrix
    camera_matrix = np.zeros((3, 4), dtype=np.float32)
    camera_matrix[:2, :] = np.reshape(m, (2, 4))
    camera_matrix[-1,:] = [0., 0., 0., 1.]

    # return the camera matrix
    return camera_matrix

'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''
def rms_error(camera_matrix, real_XY, front_image, back_image):
    # compose real homogeneous coordinates
    real_XYZ_front = np.zeros((real_XY.shape[0], 4))
    real_XYZ_front[:,0:2] = real_XY
    real_XYZ_front[:,2] = 0.0 # z = 0
    real_XYZ_front[:,3] = 1.0 # addition of 4th dimension in homogeneous coordinate system

    real_XYZ_back = np.zeros((real_XY.shape[0], 4))
    real_XYZ_back[:,0:2] = real_XY
    real_XYZ_back[:,2] = 150.0 # z = 150
    real_XYZ_back[:,3] = 1.0 # addition of 4th dimension in homogeneous coordinate system

    # put together the real coordinates
    real_XYZ_homogeneous = np.vstack((real_XYZ_front, real_XYZ_back))

    # put together the ground-truth corresponding image projection coordinates
    projected_XY_gt = np.vstack((front_image, back_image))

    # project the points onto image plane using camera_matrix
    projected_XY = np.transpose(np.dot(camera_matrix, np.transpose(real_XYZ_homogeneous)))

    # compute RMS Error
    rmse = 0.0
    for n_pt in range(projected_XY_gt.shape[0]):
        rmse += np.power((projected_XY_gt[n_pt, 0] - projected_XY[n_pt, 0]), 2) + \
                np.power((projected_XY_gt[n_pt, 1] - projected_XY[n_pt, 1]), 2)
    # take mean
    rmse /= projected_XY_gt.shape[0]
    rmse = np.sqrt(rmse)

    # # Visualize the projection using camera matrix
    # front_img = cv2.imread('front.png', cv2.IMREAD_COLOR)
    # for pt in projected_XY[:int(projected_XY.shape[0]/2), :]:
    #     cv2.circle(front_img, (int(pt[0]), int(pt[1])), radius=4, thickness=2, color=(0,0,255))
    # cv2.imshow('front projected image', front_img)

    # return rms error
    return rmse

if __name__ == '__main__':
    # Loading the example coordinates setup
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    print("Camera Matrix:\n", camera_matrix)
    print()
    print("RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image))

    # ## visualize points
    # # read front image
    # front_img = cv2.imread('front.png', cv2.IMREAD_COLOR)
    # for pt in front_image:
    #     cv2.circle(front_img, (int(pt[0]), int(pt[1])), radius=4, thickness=2, color=(0,0,255))
    # cv2.imshow('front image', front_img)
    # # wait for a key to be pressed
    # cv2.waitKey(0)