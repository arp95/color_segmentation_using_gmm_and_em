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


# set data path
k = 6
d = 3
iterations = 500
args = sys.argv
video_path = ""
file_path = ""
if(len(args) > 2):
    video_path = args[1]
    file_path = args[2]

# get training data
training_data = get_training_data(file_path, 1, 1, 1)
training_data = training_data[:20000, :]

# get the weights, mean and variances of gaussian
(weights_gaussian, mean_gaussian, covariance_matrix_gaussian) = run_expectation_maximization_algorithm(training_data.shape[0], d, k, iterations, training_data)

print(weights_gaussian)
print(mean_gaussian)
print(covariance_matrix_gaussian)

# code for segmenting the buoy
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('yellow_buoy_3D_gauss.avi', fourcc, 5.0, (640, 480))
frame_count = 0
while (cap.isOpened()):
    success, frame = cap.read()
    if success == False:
        break    

    # steps to find the probability of each pixel in the k-gaussians
    img = np.reshape(frame, (frame.shape[0] * frame.shape[1], d))
    prob = np.zeros((frame.shape[0] * frame.shape[1], k))
    likelihood = np.zeros((frame.shape[0] * frame.shape[1], k))
    for index in range(0, k):
        prob[: ,index:index+1] = gaussian_estimation_3d(img, mean_gaussian[index], covariance_matrix_gaussian[index]) * weights_gaussian[index]
        likelihood = prob.sum(1)
    prob = np.reshape(likelihood, (frame.shape[0], frame.shape[1]))
    prob[prob > np.max(prob) / 9.0] = 255
    
    # pre-process image and create a binary image
    output_image = np.zeros_like(frame)
    output_image[:, :, 0] = prob
    output_image[:, :, 1] = prob
    output_image[:, :, 2] = prob
    gray_output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    output_image = cv2.GaussianBlur(gray_output_image, (3, 3), 2)
    _, edged = cv2.threshold(output_image, 150, 255, 0)
    kernel2 = np.ones((3,3), np.uint8)
    edged = cv2.dilate(edged, kernel2, iterations = 6)
    if(frame_count < 100):
        for row in range(0, frame.shape[0] / 2):
            for col in range(0, frame.shape[1]):
                edged[row, col] = 0    

    # find contours and segment the orange buoy
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    x_min = 1000
    optimal_x = -1
    optimal_y = -1
    optimal_radius = -1
    for contour in cnts:
        if(cv2.contourArea(contour) > 100):
            (x,y), radius = cv2.minEnclosingCircle(contour)
            if(radius > 5 and radius < 50):
                if(int(x) < x_min):
                    x_min = x
                    optimal_x = x
                    optimal_y = y
                    optimal_radius = radius
    if(optimal_radius != -1):
        cv2.circle(frame, (int(optimal_x), int(optimal_y)), int(optimal_radius), (0, 255, 255), 4)
    out.write(frame)
    frame_count = frame_count + 1
