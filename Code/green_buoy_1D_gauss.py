"""
 *  MIT License
 *
 *  Copyright (c) 2019 Arpit Aggarwal Shantam Bajpai
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without
 *  limitation the rights to use, copy, modify, merge, publish, distribute,
 *  sublicense, and/or sell copies of the Software, and to permit persons to
 *  whom the Software is furnished to do so, subject to the following
 *  conditions:
 *
 *  The above copyright notice and this permission notice shall be included
 *  in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
"""


# header files
from utils import *
import sys


# set data path, dimension of gaussian, number of gaussians and number of iterations
k = 2
d = 1
iterations = 500
args = sys.argv
video_path = ""
file_path = ""
if(len(args) > 2):
    video_path = args[1]
    file_path = args[2]

# get training data
training_data = get_training_data(file_path, 0, 1, 0)
training_data = training_data[:20000, :]

# get the weights, mean and variances of gaussian
(weights_gaussian, mean_gaussian, covariance_matrix_gaussian) = run_expectation_maximization_algorithm(training_data.shape[0], d, k, iterations, training_data)

print(weights_gaussian)
print(mean_gaussian)
print(covariance_matrix_gaussian)

# segmenting the buoy
frame_count = 0
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('green_buoy_1D_gauss.avi', fourcc, 5.0, (640, 480))
while (cap.isOpened()):
    success, frame = cap.read()
    if success == False:
        break    

    # steps to find the probability of each pixel in the k-gaussians
    img = np.reshape(frame[:, :, 1], (frame.shape[0] * frame.shape[1], d))
    prob = np.zeros((frame.shape[0] * frame.shape[1], k))
    likelihood = np.zeros((frame.shape[0] * frame.shape[1], k))
    for index in range(0, k):
        prob[: ,index:index+1] = gaussian_estimation_array(img, mean_gaussian[index], covariance_matrix_gaussian[index], 1) * weights_gaussian[index]
        likelihood = prob.sum(1)
    prob = np.reshape(likelihood, (frame.shape[0], frame.shape[1]))
    prob[prob > np.max(prob) / 2.0] = 255
    
    # pre-process image and create a binary image
    output_image = np.zeros_like(frame)
    output_image[:, :, 1] = prob
    gray_output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    output_image = cv2.GaussianBlur(gray_output_image, (5, 5), 5)
    _, edged = cv2.threshold(output_image, 125, 255, 0)
    kernel2 = np.ones((2, 2), np.uint8)
    edged = cv2.dilate(edged, kernel2, iterations = 6)
    for row in range(0, frame.shape[0] / 2):
        for col in range(0, frame.shape[1]):
            edged[row, col] = 0
    if(frame_count < 35):
        for row in range(0, frame.shape[0]):
            for col in range(frame.shape[1] - 200, frame.shape[1]):
                edged[row, col] = 0

    # find contours and segment the green buoy
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    x_max = 0
    optimal_x = -1
    optimal_y = -1
    optimal_radius = -1
    for contour in cnts:
        if(cv2.contourArea(contour) > 150):
            (x,y), radius = cv2.minEnclosingCircle(contour)
            if(radius > 10 and radius < 50):
                if(int(x) > x_max and int(x) > 150):
                    x_max = x
                    optimal_x = x
                    optimal_y = y
                    optimal_radius = radius
    if(optimal_radius != -1 and frame_count < 50):
        cv2.circle(frame, (int(optimal_x), int(optimal_y)), int(optimal_radius), (0, 255, 0), 5)
    out.write(frame)
    frame_count = frame_count + 1
