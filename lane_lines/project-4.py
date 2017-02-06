import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from scipy.signal import convolve

try:
    with open('calibration.p', 'rb') as f:
        ret, mtx, dist = pickle.load(f)
except:
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    # camera calibration
    for i in os.listdir('camera_cal'):

        image = plt.imread('camera_cal/{}'.format(i))
        imshp = image.shape

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # search for chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6))

        if not ret:
            print('didnt find corners {}'.format(i))
            continue

        # If found, add object points, image points
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(image, (9, 6), corners, ret)
        # write_name = 'corners_found'+str(idx)+'.jpg'
        # cv2.imwrite(write_name, img)
        print(i)
        plt.imshow(image)
        plt.show()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        dst = cv2.undistort(image, mtx, dist, None, mtx)
        cv2.imwrite('output_images/{}'.format(i), dst)

    with open('calibration.p', 'wb') as f:
        pickle.dump((ret, mtx, dist), f)


def peaks(inp, index0, index1):
    inp = (np.conjugate(inp) * inp).real
    max0 = np.argmax(inp[index0:index1])
    max1 = np.argsort(inp[index0:index1])[::-1]
    max1 = max1[(15 < abs(max1 - max0)) & (abs(max1 - max0) < 70)]
    return np.average((max1[0], max0)).astype(np.int32)

for i in os.listdir('test_images'):
    image = plt.imread('test_images/{}'.format(i))
    img_size = (image.shape[1], image.shape[0])

    undistorted = cv2.undistort(image, mtx, dist, None, mtx)


    # vertices = [[  505.   490.], [  800.   490.], [ 1060.   610.], [  250.   610.]]
    src = np.float32([((img_size[0] - 270) / 2, (img_size[1] + 260) / 2),
                      ((img_size[0] + 320) / 2, (img_size[1] + 260) / 2),
                      (img_size[0] - 220, img_size[1] - 110),
                      (250, img_size[1] - 110)])

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    offset = 100
    dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                      [img_size[0] - offset, img_size[1] - offset],
                      [offset, img_size[1] - offset]])
    # Given src and dst points, calculate the perspective transform matrix
    print(src, dst, img_size)
    #plt.imshow(undistorted)
    #plt.show()
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()

    warped = cv2.warpPerspective(undistorted, M, img_size)

    #plt.imshow(warped)
    #plt.show()

    R = warped[:, :, 0]
    R_thresh = (200, 255)

    hls = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    S_thresh = (100, 255)
    H = hls[:, :, 0]
    H_thresh = (15, 100)
    mask = np.zeros_like(S)
    mask[((S >= S_thresh[0]) & (S <= S_thresh[1])) |
         ((R >= R_thresh[0]) & (R <= R_thresh[1])) |
         ((H >= H_thresh[0]) & (H <= H_thresh[1]))] = 1
    combined_color = mask

    sobelx = cv2.Sobel(combined_color, cv2.CV_64F, 1, 0, ksize=9)
    sobely = cv2.Sobel(combined_color, cv2.CV_64F, 0, 1, ksize=9)
    # 3) Take the absolute value of the x and y gradients
    x = np.absolute(sobelx)
    y = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    g = np.arctan2(y, x)
    # 5) Create a binary mask where direction thresholds are met
    mask = np.zeros_like(combined_color)
    sobel_thresh = (0.8, 1.2)
    mask[(g >= sobel_thresh[0]) & (g <= sobel_thresh[1])] = 1

    #plt.imshow(mask)
    #plt.show()

    # Take a histogram of the bottom half of the image
    histogram = np.sum(mask[int(mask.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((mask, mask, mask)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    ker = np.concatenate((np.exp((-np.abs((np.arange(-25, 25, 1)) / 9))),
                          np.exp((-np.abs((np.arange(-25, 25, 1)) / 9)))), axis=0)[16:-16]
    deconvlolution = convolve(histogram[:midpoint], ker, mode='same')
    out = deconvlolution
    # plt.plot(histogram[:midpoint])
    # plt.show()
    # plt.plot(out)
    # plt.show()
    leftx_base = peaks(out, 150, 510) + 150

    deconvlolution = convolve(histogram[midpoint:], ker, mode='same')
    out = deconvlolution
    rightx_base = peaks(out, 150, 510) + midpoint + 150

    # plt.plot(histogram[midpoint:])
    # plt.show()
    # plt.plot(out)
    # plt.show()

    # Choose the number of sliding windows
    nwindows = 7
    # Set height of windows
    window_height = np.int(mask.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = mask
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = len(ker) + 10

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = ([], [])
    right_lane_inds = ([], [])

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = mask.shape[0] - (window + 1) * window_height
        win_y_high = mask.shape[0] - window * window_height
        y = (win_y_high + win_y_low) / 2

        deconvlolution = convolve(np.sum(mask[win_y_low:win_y_high,
                                         leftx_current - margin:leftx_current + margin],
                                         axis=0), ker, mode='same')
        # plt.plot(deconvlolution)
        # plt.show()
        left = deconvlolution
        new_left_candidate = leftx_current + (peaks(left, 0, 2 * margin) - margin)
        win_xleft_low = new_left_candidate - margin
        win_xleft_high = new_left_candidate + margin

        deconvlolution = convolve(np.sum(mask[win_y_low:win_y_high,
                                         rightx_current - margin:rightx_current + margin],
                                         axis=0), ker, mode='same')
        right = deconvlolution
        # plt.plot(deconvlolution)
        # plt.show()
        new_right_candidate = rightx_current + (peaks(right, 0, 2 * margin) - margin)
        win_xright_low = new_right_candidate - margin
        win_xright_high = new_right_candidate + margin
        # Draw the windows on the visualization image

        # Identify the nonzero pixels in x and y within the window
        # good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        # nonzerox < win_xleft_high)).nonzero()[0]
        # good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        # nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists


        # If you found > minpix pixels, recenter next window on their mean position
        if abs(new_left_candidate - leftx_current) < (margin / 3):
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            leftx_current = new_left_candidate
            left_lane_inds[0].append(new_left_candidate)
        else:
            left_lane_inds[0].append(leftx_current)
        left_lane_inds[1].append(y)

        if abs(new_right_candidate - rightx_current) < (margin / 3):
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            rightx_current = new_right_candidate
            right_lane_inds[0].append(new_right_candidate)
        else:
            right_lane_inds[0].append(rightx_current)
        right_lane_inds[1].append(y)


    # Concatenate the arrays of indices
    # left_lane_inds = np.concatenate(left_lane_inds)
    # right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = left_lane_inds[0]
    lefty = left_lane_inds[1]
    rightx = right_lane_inds[0]
    righty = right_lane_inds[1]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, mask.shape[0]-1, mask.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    plt.show()

    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3 / 173  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 767  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(np.float32(lefty)*ym_per_pix, np.float32(leftx)*xm_per_pix, 2)
    right_fit_cr = np.polyfit(np.float32(righty)*ym_per_pix, np.float32(rightx)*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    new = np.zeros_like(warped)
    points = np.hstack((np.array([np.transpose(np.vstack([left_fitx, ploty]))]),
                       np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])))
    cv2.fillPoly(new, points.astype(np.int32), (0, 255, 0))

    M_in = cv2.getPerspectiveTransform(dst, src)
    unwarp = cv2.warpPerspective(new, M_in, img_size)

    # vertices = [[  505.   490.], [  800.   490.], [ 1060.   610.], [  250.   610.]]
    fig = plt.figure(frameon=False)
    plt.imshow(image)
    plt.imshow(unwarp, alpha=0.5)
    plt.show()



# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


from moviepy.editor import VideoFileClip
