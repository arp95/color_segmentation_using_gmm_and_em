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
k1 = 4 #(green)
k2 = 6 #(yellow)
k3 = 6 #(orange)
d = 3
iterations = 500
args = sys.argv
video_path = ""
file_path = ""
if(len(args) > 4):
    video_path = args[1]
    file_path1 = args[2]
    file_path2 = args[3]
    file_path3 = args[4]

# get training data
training_data1 = get_training_data(file_path1, 1, 1, 1)
training_data1 = training_data1[:20000, :]
training_data2 = get_training_data(file_path2, 1, 1, 1)
training_data2 = training_data2[:20000, :]
training_data3 = get_training_data(file_path3, 1, 1, 1)
training_data3 = training_data3[:20000, :]

# get the weights, mean and variances of gaussian
(weights_gaussian1, mean_gaussian1, covariance_matrix_gaussian1) = run_expectation_maximization_algorithm(training_data1.shape[0], d, k1, iterations, training_data1)
(weights_gaussian2, mean_gaussian2, covariance_matrix_gaussian2) = run_expectation_maximization_algorithm(training_data2.shape[0], d, k2, iterations, training_data2)
(weights_gaussian3, mean_gaussian3, covariance_matrix_gaussian3) = run_expectation_maximization_algorithm(training_data3.shape[0], d, k3, iterations, training_data3)
print("Found the parameters for the GMM!")

# code for segmenting the buoy
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('all_buoy_3D_gauss.avi', fourcc, 5.0, (640, 480))
frame_count = 0
hashmap = {}
while (cap.isOpened()):
    success, frame = cap.read()
    if success == False:
        break    

    # steps to find the probability of each pixel in the k-gaussians
    img = np.reshape(frame, (frame.shape[0] * frame.shape[1], d))
    prob1 = np.zeros((frame.shape[0] * frame.shape[1], k1))
    likelihood1 = np.zeros((frame.shape[0] * frame.shape[1], k1))
    for index in range(0, k1):
        prob1[: ,index:index+1] = gaussian_estimation_3d(img, mean_gaussian1[index], covariance_matrix_gaussian1[index]) * weights_gaussian1[index]
        likelihood1 = prob1.sum(1)
    prob1 = np.reshape(likelihood1, (frame.shape[0], frame.shape[1]))
    prob1[prob1 > np.max(prob1) / 10] = 255
    prob2 = np.zeros((frame.shape[0] * frame.shape[1], k2))
    likelihood2 = np.zeros((frame.shape[0] * frame.shape[1], k2))
    for index in range(0, k2):
        prob2[: ,index:index+1] = gaussian_estimation_3d(img, mean_gaussian2[index], covariance_matrix_gaussian2[index]) * weights_gaussian2[index]
        likelihood2 = prob2.sum(1)
    prob2 = np.reshape(likelihood2, (frame.shape[0], frame.shape[1]))
    prob2[prob2 > np.max(prob2) / 9.0] = 255
    prob3 = np.zeros((frame.shape[0] * frame.shape[1], k3))
    likelihood3 = np.zeros((frame.shape[0] * frame.shape[1], k3))
    for index in range(0, k3):
        prob3[: ,index:index+1] = gaussian_estimation_3d(img, mean_gaussian3[index], covariance_matrix_gaussian3[index]) * weights_gaussian3[index]
        likelihood3 = prob3.sum(1)
    prob3 = np.reshape(likelihood3, (frame.shape[0], frame.shape[1]))
    prob3[prob3 > np.max(prob3) / 8.0] = 255
    
    # pre-process image and create a binary image
    output_image1 = np.zeros_like(frame)
    output_image1[:, :, 0] = prob1
    output_image1[:, :, 1] = prob1
    output_image1[:, :, 2] = prob1
    gray_output_image1 = cv2.cvtColor(output_image1, cv2.COLOR_BGR2GRAY)
    output_image1 = cv2.GaussianBlur(gray_output_image1, (5, 5), 5)
    edged1 = cv2.Canny(output_image1, 30, 255)
    _, edged1 = cv2.threshold(output_image1, 125, 255, 0)
    kernel2 = np.ones((2, 2), np.uint8)
    edged1 = cv2.dilate(edged1, kernel2, iterations = 6)
    for row in range(0, frame.shape[0] / 2):
        for col in range(0, frame.shape[1]):
            edged1[row, col] = 0
    if(frame_count < 35):
        for row in range(0, frame.shape[0]):
            for col in range(frame.shape[1] - 200, frame.shape[1]):
                edged1[row, col] = 0

    output_image2 = np.zeros_like(frame)
    output_image2[:, :, 0] = prob2
    output_image2[:, :, 1] = prob2
    output_image2[:, :, 2] = prob2
    gray_output_image2 = cv2.cvtColor(output_image2, cv2.COLOR_BGR2GRAY)
    output_image2 = cv2.GaussianBlur(gray_output_image2, (3, 3), 2)
    _, edged2 = cv2.threshold(output_image2, 150, 255, 0)
    kernel2 = np.ones((3,3), np.uint8)
    edged2 = cv2.dilate(edged2, kernel2, iterations = 6)
    if(frame_count < 100):
        for row in range(0, frame.shape[0] / 2):
            for col in range(0, frame.shape[1]):
                edged2[row, col] = 0   

    output_image3 = np.zeros_like(frame)
    output_image3[:, :, 0] = prob3
    output_image3[:, :, 1] = prob3
    output_image3[:, :, 2] = prob3
    gray_output_image3 = cv2.cvtColor(output_image3, cv2.COLOR_BGR2GRAY)
    output_image3 = cv2.GaussianBlur(gray_output_image3, (5, 5), 5)
    _, edged3 = cv2.threshold(output_image3, 30, 255, 0)

    # find contours and segment the green buoy
    (cnts, _) = cv2.findContours(edged1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
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
    if(optimal_radius != -1 and frame_count < 46):
        cv2.circle(frame, (int(optimal_x), int(optimal_y)), int(optimal_radius), (0, 255, 0), 5)
 
    (cnts, _) = cv2.findContours(edged2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
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
        cv2.circle(frame, (int(optimal_x), int(optimal_y)), int(optimal_radius), (0, 255, 255), 5)

    (cnts, _) = cv2.findContours(edged3.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    radius = 0
    if(len(cnts) > 0):
        cnts, _ = sort_contours(cnts, method="right-to-left")
        hull = cv2.convexHull(cnts[0])
        (x, y), radius = cv2.minEnclosingCircle(hull)
        hashmap[frame_count] = ((x, y), radius)
    if(radius < 9 and hashmap.get(frame_count - 1)):
        (x, y), radius = hashmap[frame_count - 1]
    if radius > 9:
        cv2.circle(frame, (int(x), int(y)), int(radius + 1), (0, 165, 255), 5)
    cv2.imshow("Image", frame)
    cv2.waitKey(0)
    out.write(frame)
    frame_count = frame_count + 1
