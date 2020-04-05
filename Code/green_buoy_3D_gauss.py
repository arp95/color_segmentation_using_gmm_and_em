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
k = 4
d = 3
iterations = 180
args = sys.argv
video_path = ""
file_path = ""
if(len(args) > 2):
    video_path = args[1]
    file_path = args[2]

"""
# get training data
training_data = get_training_data(file_path, 1, 1, 1)
training_data = training_data[:20000, :]

# get the weights, mean and variances of gaussian
(weights_gaussian, mean_gaussian, covariance_matrix_gaussian) = run_expectation_maximization_algorithm(training_data.shape[0], d, k, iterations, training_data)

print(weights_gaussian)
print(mean_gaussian)
print(covariance_matrix_gaussian)
"""
weights_gaussian = np.array([0.18629567531089725, 0.20251932184135432, 0.40918125362489594, 0.2020037492228503])
mean_gaussian = np.array([[189.2024833 , 245.14785638, 218.92626451],
 [139.85593966, 243.19534678, 170.7373105 ],
 [114.65267232, 192.30751732, 108.84220896],
 [125.42810692, 226.39760109, 138.04298968]])
covariance_matrix_gaussian = np.array([[[1516.30024442,  199.03687814,  783.13729682],
       [ 199.03687814,   93.56072027,  153.12934751],
       [ 783.13729682,  153.12934751,  544.7206388 ]],
 [[ 48.00150021,   6.85113533,  71.62070739],
       [  6.85113533,  15.18477423,  12.73707837],
       [ 71.62070739,  12.73707837, 172.43460924]],
 [[ 70.39419933,  64.87552097,  75.58740199],
       [ 64.87552097, 246.15377109, 135.87945142],
       [ 75.58740199, 135.87945142, 139.18688082]],
 [[23.63243822,  7.30339784, 17.04521036],
       [ 7.30339784, 61.44401099, 46.086562  ],
       [17.04521036, 46.086562  , 80.98872277]]])


# code for segmenting the buoy
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('green_buoy_3D_gauss.avi', fourcc, 5.0, (640, 480))
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
    prob[prob > np.max(prob) / 8.5] = 255
    
    # pre-process image and create a binary image
    output_image = np.zeros_like(frame)
    output_image[:, :, 0] = prob
    output_image[:, :, 1] = prob
    output_image[:, :, 2] = prob
    gray_output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    output_image = cv2.GaussianBlur(gray_output_image, (3, 3), 5)
    edged = cv2.Canny(output_image, 50, 255)
    
    # find contours and segment the orange buoy
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cnts, _ = sort_contours(cnts, method="right-to-left")
    hull = cv2.convexHull(cnts[0])
    (x, y), radius = cv2.minEnclosingCircle(hull)
    if radius > 5:
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        out.write(frame)
    else:
        out.write(frame)
