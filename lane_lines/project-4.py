import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from scipy.signal import convolve

try:
    # if a calibration pickle was found, read out the parameters. otherwise calibrate images
    with open('calibration.p', 'rb') as f:
        ret, mtx, dist = pickle.load(f)
except:
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    # camera calibration for failes in the camera_cal relative folder path
    for i in os.listdir('camera_cal'):

        image = plt.imread('camera_cal/{}'.format(i))
        imshp = image.shape

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # search for chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6))

        if not ret:
            print('didnt find corners for {}'.format(i))
            continue

        # If found, add object points, image points
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(image, (9, 6), corners, ret)

        # calibrate using object and image points collected
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # undistort and save images to output_images
        dst = cv2.undistort(image, mtx, dist, None, mtx)
        cv2.imwrite('output_images/{}'.format(i), dst)

    # pickle parameters for futute runs
    with open('calibration.p', 'wb') as f:
        pickle.dump((ret, mtx, dist), f)


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self, n=6):
        # number of images to average over
        self.n = n
        # hold the fitted polynomial coefficients [A, B, C] for the last n entries
        self.left_params = []
        self.right_params = []

        # store radius of curvature calculated for the left and right fits for the last n entries
        self.curv_left = []
        self.curv_right = []

        # previous base position for left and right lines
        self.base_leftx = 0
        self.base_rightx = 0


def peaks(inp, index0, index1):
    # return the average of the first two peaks found in the convolution
    # within index0 and index1. if not found, return the average of the two indexes
    inp = (np.conjugate(inp) * inp).real
    max1 = np.argsort(inp[index0:index1])[::-1]
    max0 = max1[0] if len(max1) else int((index0 + index1) / 2)
    max1 = max1[(15 < abs(max1 - int(max0))) & (abs(max1 - int(max0)) < 50)]
    return np.average((max1[0] if len(max1) else int((index0 + index1) / 2), max0)).astype(np.int32)


def process_image(image):
    img_size = (image.shape[1], image.shape[0])

    undistorted = cv2.undistort(image, mtx, dist, None, mtx)

    # create a red colour threshold mask
    R = undistorted[:, :, 0]
    R_thresh = (225, 255)

    hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
    # create a saturation threshold mask
    S = hls[:, :, 2]
    S_thresh = (175, 195)
    # create a lightness threshold mask
    L = hls[:, :, 1]
    L_thresh = (215, 255)
    # create a hue thresholded mask
    H = hls[:, :, 0]
    H_thresh = (35, 75)

    # find the sobel x and y gradients to create a threshold direction mask
    mask = np.zeros_like(S)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    # Take the absolute value of the x and y gradients
    x = np.absolute(sobelx)
    y = np.absolute(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    g = np.arctan2(y, x)
    # Create a binary mask where direction thresholds are met
    sobel_thresh = (0.7, 1.3)

    # combine all the masks such that any of the colour space masks are complied with
    # and the direction of gradient mask condition is also true
    mask[(((S >= S_thresh[0]) & (S <= S_thresh[1])) |
         ((R >= R_thresh[0]) & (R <= R_thresh[1])) |
         ((H >= H_thresh[0]) & (H <= H_thresh[1])) |
          (L >= L_thresh[0]) & (L <= L_thresh[1])) &
         (g >= sobel_thresh[0]) & (g <= sobel_thresh[1])] = 1

    # visualise the colour binary mask here if necessary
    # plt.imshow(mask)
    # plt.show()

    # define corners of the lane lines to be warped to birds-eye view
    # vertices = [[515.  475.], [765.  475.], [1280.  720.], [0.  720.]]
    src = np.float32([((img_size[0] - 250) / 2, (img_size[1] + 230) / 2),
                      ((img_size[0] + 250) / 2, (img_size[1] + 230) / 2),
                      (img_size[0], img_size[1]),
                      (0, img_size[1])])

    # draw lines showing the borders of the src points; only done to diagnose
    # cv2.line(undistorted, tuple(src[0]), tuple(src[1]), (255, 0, 0), thickness=3)
    # cv2.line(undistorted, tuple(src[1]), tuple(src[2]), (255, 0, 0), thickness=3)
    # cv2.line(undistorted, tuple(src[2]), tuple(src[3]), (255, 0, 0), thickness=3)
    # cv2.line(undistorted, tuple(src[3]), tuple(src[0]), (255, 0, 0), thickness=3)

    # no offset on the edges of the image and dst is basically the four corners of the image
    offset = 0
    dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                      [img_size[0] - offset, img_size[1] - offset],
                      [offset, img_size[1] - offset]])
    # Given src and dst points, calculate the perspective transform matrix
    # print(src, dst, img_size)
    # visualise before warping
    # plt.imshow(mask)
    # plt.show()

    # construct transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(mask, M, img_size)

    # visualise warped image here if necessary
    # plt.imshow(warped)
    # plt.show()

    # create a mirrored lorentzian kernel to filter out the lane lines out of the image pixels
    ker = np.concatenate((np.exp((-np.abs((np.arange(-13, 13, 1)) / 7))),
                          np.exp((-np.abs((np.arange(-13, 13, 1)) / 7)))), axis=0)[6:-6]

    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped[int((warped.shape[0] + 400) / 2):, :], axis=0)

    # Create an output image to draw on and  visualize the result, if necessary
    # out_img = np.dstack((warped, warped, warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)

    # perform a integral based on the kernel to find points where the kernel
    # is best observed within the histogram
    deconvlolution = convolve(histogram[:midpoint], ker, mode='same')
    out1 = deconvlolution

    # visualise the histogram and deconvolution here if necessary
    # plt.plot(histogram[:midpoint])
    # plt.show()
    # plt.plot(out1)
    # plt.show()
    leftx_current = peaks(out1, 20, 620) + 20

    # same as above for the right side
    deconvlolution = convolve(histogram[midpoint:], ker, mode='same')
    out = deconvlolution

    # visualise the histogram and deconvolution here if necessary
    # plt.plot(histogram[midpoint:])
    # plt.show()
    # plt.plot(out)
    # plt.show()
    rightx_current = peaks(out, 20, 620) + midpoint + 20

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(warped.shape[0] / nwindows)

    # stored base positions to be updated for each side if the changes
    # are smaller than 50 pixels from the last image
    if line.base_leftx and abs(line.base_leftx - leftx_current) < 70:
        leftx_current = line.base_leftx
    else:
        line.base_leftx = leftx_current

    if line.base_rightx and abs(line.base_rightx - rightx_current) < 70:
        rightx_current = line.base_rightx
    else:
        line.base_rightx = rightx_current

    # Set the width of the windows +/- margin
    margin = len(ker) + 30

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = ([], [])
    right_lane_inds = ([], [])

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window + 1) * window_height
        win_y_high = warped.shape[0] - window * window_height
        y = (win_y_high + win_y_low) / 2

        # perform a integral based on the kernel to find points where the kernel
        # is best observed within the window
        deconvlolution = convolve(np.sum(warped[win_y_low:win_y_high,
                                         leftx_current - margin:leftx_current + margin],
                                         axis=0), ker, mode='same')
        # visualise here if necessary
        # plt.plot(deconvlolution)
        # plt.show()
        left = deconvlolution
        # find and store new candidate
        new_left_candidate = leftx_current + (peaks(left, 0, 2 * margin) - margin)

        # window coordinates if they are being drawn
        # win_xleft_low = new_left_candidate - margin
        # win_xleft_high = new_left_candidate + margin

        # perform a integral based on the kernel to find points where the kernel
        # is best observed within the window
        deconvlolution = convolve(np.sum(warped[win_y_low:win_y_high,
                                         rightx_current - margin:rightx_current + margin],
                                         axis=0), ker, mode='same')
        # plt.plot(deconvlolution)
        # plt.show()
        right = deconvlolution
        # find and store new candidate
        new_right_candidate = rightx_current + (peaks(right, 0, 2 * margin) - margin)

        # window coordinates if they are being drawn
        # win_xright_low = new_right_candidate - margin
        # win_xright_high = new_right_candidate + margin

        # if we found new candidates that arent outside the margin
        # recenter next window on their mean position and draw the windows
        # otherwise use the previous window coordinates as the current
        if abs(new_left_candidate - leftx_current) < (margin / 1.6):
            # Draw the windows on the visualization image
            # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            leftx_current = new_left_candidate
            left_lane_inds[0].append(new_left_candidate)
        else:
            left_lane_inds[0].append(leftx_current)
        left_lane_inds[1].append(y)

        if abs(new_right_candidate - rightx_current) < (margin / 1.6):
            # Draw the windows on the visualization image
            # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            rightx_current = new_right_candidate
            right_lane_inds[0].append(new_right_candidate)
        else:
            right_lane_inds[0].append(rightx_current)
        right_lane_inds[1].append(y)

    # Extract left and right line pixel positions
    leftx = left_lane_inds[0]
    lefty = left_lane_inds[1]
    rightx = right_lane_inds[0]
    righty = right_lane_inds[1]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3 / 173  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 767  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(np.float32(lefty) * ym_per_pix, np.float32(leftx) * xm_per_pix, 2)
    right_fit_cr = np.polyfit(np.float32(righty) * ym_per_pix, np.float32(rightx) * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_grad = (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])
    left_curverad = ((1 + left_grad ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_grad = (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])
    right_curverad = ((1 + right_grad ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters

    # if left plot parameters are not found, use current
    if not len(line.left_params):
        line.left_params = [left_fit, left_fit]

    # if left curvature is not found, use current
    if not len(line.curv_left):
        line.curv_left = [left_curverad, left_curverad]

    # if there is some curvature data stored, check the new curvature is not too far off
    elif abs(left_curverad - np.average(line.curv_left)) < 600:
        # if we have store above line.n times for curvature and fit parameters
        # pop the first in the stack
        if len(line.curv_left) > line.n:
            line.curv_left.pop(0)
        if len(line.left_params) > line.n:
            line.left_params.pop(0)
        # store the new values
        line.left_params.append(left_fit)
        line.curv_left.append(left_curverad)
    # calculate the average of the stored curvature and fit parameter values
    left_curverad = np.average(line.curv_left)
    left_fit = np.average(line.left_params, axis=0)
    # use the average fit parameters to plot line for the overlay
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

    # if left plot parameters are not found, use current
    if not len(line.right_params):
        line.right_params = [right_fit, right_fit]

    # if left curvature is not found, use current
    if not len(line.curv_right):
        line.curv_right = [right_curverad, right_curverad]

    # if there is some curvature data stored, check the new curvature is not too far off
    elif abs(right_curverad - np.average(line.curv_right)) < 600:
        # if we have store above line.n times for curvature and fit parameters
        # pop the first in the stack
        if len(line.curv_right) > line.n:
            line.curv_right.pop(0)
        if len(line.right_params) > line.n:
            line.right_params.pop(0)
        # store the new values
        line.curv_right.append(right_curverad)
        line.right_params.append(right_fit)
    # calculate the average of the stored curvature and fit parameter values
    right_curverad = np.average(line.curv_right)
    right_fit = np.average(line.right_params, axis=0)
    # use the average fit parameters to plot line for the overlay
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # visualise the plotted windows and poly fit here if necessary
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    #
    # print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    # plt.show()

    # create new image and draw the new fitted lines as an overlay
    new = cv2.cvtColor(np.zeros_like(warped), cv2.COLOR_GRAY2RGB)
    points = np.hstack((np.array([np.transpose(np.vstack([left_fitx, ploty]))]),
                        np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])))
    cv2.fillPoly(new, points.astype(np.int32), (0, 255, 0))

    # contruct the inverse transform matrix for warping
    M_inv = cv2.getPerspectiveTransform(dst, src)
    # warp the image
    unwarp = cv2.warpPerspective(new, M_inv, img_size)

    # draw the overlay with alpha = 1, beta = 0.4 and gamma=0
    overlay = cv2.addWeighted(undistorted, 1, unwarp, 0.4, 0)

    # calculate the distance of the image centre from the middle of the lines
    # negative values represent distances to the left of centre
    car_centre = (((img_size[0])/ 2) - ((left_fitx[0] + right_fitx[0]) / 2)) * xm_per_pix

    # if the curvature values found are too high, usually means correct lines were not found
    # do not use curvature values that are too low or too high
    if 100 < abs(left_curverad) and 100 < abs(right_curverad):
        curve = ((len(left_lane_inds) * left_curverad) + (len(right_lane_inds) * right_curverad)) / (len(right_lane_inds) + (len(left_lane_inds)))
    elif 100 < abs(left_curverad) and 100 > abs(right_curverad):
        curve = left_curverad
    elif 100 > abs(left_curverad) and 100 < abs(right_curverad):
        curve = right_curverad
    else:
        curve = 100.0

    # create text on the image with the curvature and position from the centre
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, 'Curvature {:.1f}m   Position {:.2f}m'.format(curve, car_centre),
                (150, 100), font, 1, (255, 255, 255), 2)
    # visualize if necessary
    # plt.imshow(overlay)
    # plt.show()
    return overlay


for i in os.listdir('test_images'):
    img = plt.imread('test_images/{}'.format(i))
    line = Line()

    cv2.imwrite('output_images/{}'.format(i), process_image(img))


from moviepy.editor import VideoFileClip
line = Line()
output = 'processed_project_video.mp4'
clip2 = VideoFileClip('project_video.mp4')
clip = clip2.fl_image(process_image)
clip.write_videofile(output, audio=False)
line = Line()
output = 'processed_challenge_video.mp4'
clip2 = VideoFileClip('challenge_video.mp4')
clip = clip2.fl_image(process_image)
clip.write_videofile(output, audio=False)
line = Line()
output = 'processed_harder_challenge_video.mp4'
clip2 = VideoFileClip('harder_challenge_video.mp4')
clip = clip2.fl_image(process_image)
clip.write_videofile(output, audio=False)