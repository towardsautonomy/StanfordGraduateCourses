import numpy as np
import cv2
# import open3d as o3d   

def draw_tracks(frame_num, frame, mask, points_prev, points_curr, color):
    """Draw the tracks and create an image.
    """
    for i, (p_prev, p_curr) in enumerate(zip(points_prev, points_curr)):
        a, b = p_curr.ravel()
        c, d = p_prev.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(
            frame, (a, b), 3, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    cv2.imwrite('frame_%d.png'%frame_num,img)
    return img


def Q5_A():
    """Code for question 5a.

    Output:
      p0, p1, p2: (N,2) list of numpy arrays representing the pixel coordinates of the
      tracked features.  Include the visualization and your answer to the
      questions in the separate PDF.
    """
    # params for ShiTomasi corner detection
    feature_params = dict(
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(75, 75),
        maxLevel=1,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01),
        flags=(cv2.OPTFLOW_LK_GET_MIN_EIGENVALS))

    # Read the frames.
    frame1 = cv2.imread('p5_data/rgb1.png')
    frame2 = cv2.imread('p5_data/rgb2.png')
    frame3 = cv2.imread('p5_data/rgb3.png')
    frames = [frame1, frame2, frame3]

    # Convert to gray images.
    old_frame = frames[0]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create some random colors for drawing
    color = np.random.randint(0, 255, (200, 3))

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame1)

    pn = []
    for i,frame in enumerate(frames[1:]):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]

            if i == 0:
                pn.append(good_old.reshape(-1, 2))
            pn.append(good_new.reshape(-1, 2))
        
        # # plot optic flow
        # draw_tracks(i, frame.copy(), mask.copy(), p0, p1, color)

        # update the old points
        old_gray = frame_gray
        p0 = p1

    return pn[0], pn[1], pn[2]


def Q5_B(p0, p1, p2, intrinsic):
    """Code for question 5b.

    Note that depth maps contain NaN values.
    Features that have NaN depth value in any of the frames should be excluded
    in the result.

    Input:
      p0, p1, p2: (N,2) numpy arrays, the results from Q2_A.
      intrinsic: (3,3) numpy array representing the camera intrinsic.

    Output:
      p0, p1, p2: (N,3) numpy arrays, the 3D positions of the tracked features
      in each frame.
    """
    depth0 = np.loadtxt('p5_data/depth1.txt')
    depth1 = np.loadtxt('p5_data/depth2.txt')
    depth2 = np.loadtxt('p5_data/depth3.txt')

    p0_3d, p1_3d, p2_3d = [], [], []
    # go through each points and get their 3D coordinates
    for pt0_2d, pt1_2d, pt2_2d in zip(p0, p1, p2):
        z0 = float(depth0[int(pt0_2d[1]), int(pt0_2d[0])])
        z1 = float(depth1[int(pt1_2d[1]), int(pt1_2d[0])])
        z2 = float(depth2[int(pt2_2d[1]), int(pt2_2d[0])])
        if (not np.isnan(z0)) and (not np.isnan(z1)) and (not np.isnan(z2)):
            x = (pt0_2d[0] - intrinsic[0,2]) * z0 / intrinsic[0,0]
            y = (pt0_2d[1] - intrinsic[1,2]) * z0 / intrinsic[1,1]
            p0_3d.append([x, y, z0])

            x = (pt1_2d[0] - intrinsic[0,2]) * z1 / intrinsic[0,0]
            y = (pt1_2d[1] - intrinsic[1,2]) * z1 / intrinsic[1,1]
            p1_3d.append([x, y, z1])

            x = (pt2_2d[0] - intrinsic[0,2]) * z2 / intrinsic[0,0]
            y = (pt2_2d[1] - intrinsic[1,2]) * z2 / intrinsic[1,1]
            p2_3d.append([x, y, z2])
    p0_3d = np.array(p0_3d, dtype=np.float64)
    p1_3d = np.array(p1_3d, dtype=np.float64)
    p2_3d = np.array(p2_3d, dtype=np.float64)

    return p0_3d, p1_3d, p2_3d

if __name__ == "__main__":
    p0, p1, p2 = Q5_A()
    intrinsic = np.array([[486, 0, 318.5],
                          [0, 491, 237],
                          [0, 0, 1]])
    p0, p1, p2 = Q5_B(p0, p1, p2, intrinsic)
    
    # # visualize using Open3D
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(p0)
    # o3d.visualization.draw_geometries([pcd])

    # pcd.points = o3d.utility.Vector3dVector(p1)
    # o3d.visualization.draw_geometries([pcd])

    # pcd.points = o3d.utility.Vector3dVector(p2)
    # o3d.visualization.draw_geometries([pcd])
    