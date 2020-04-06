# Code for Project 3 of ENPM 673-Perception for Autonomous Robots


### Authors
Shantam Bajpai
Arpit Aggarwal


### Instructions for running the code
To run the code for the files in Code folder, follow the following commands:


Running all_buoy_3D_gauss.py file, follow the following command:

```
cd Code
python all_buoy_3D_gauss.py 'video_path(in .mp4 format)' 'train_file_path_green' 'train_file_path_yellow' 'train_file_path_orange'
```
where, video_path and train_file_path_green, train_file_path_yellow and train_file_path_orange are the paths for input video and training data of the buoys. For example, running the python file on my local setup was:

```
cd Code/
python Code/all_buoy_3D_gauss.py /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/detectbuoy.avi /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/buoy3/train /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/buoy1/train /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/buoy2/train
```


1. For running yellow_buoy files:
```
cd Code
python yellow_buoy_1D_gauss.py 'video_path(in .mp4 format)' 'train_file_path'
```
where, video_path and train_file_path are the paths for input video and training data of the yellow buoy. For example, running the python file on my local setup was:

```
cd Code/
python yellow_buoy_1D_gauss.py /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/detectbuoy.avi /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/buoy1/train
```


```
cd Code
python yellow_buoy_3D_gauss.py 'video_path(in .mp4 format)' 'train_file_path'
```
where, video_path and train_file_path are the paths for input video and training data of the yellow buoy. For example, running the python file on my local setup was:

```
cd Code/
python yellow_buoy_3D_gauss.py /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/detectbuoy.avi /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/buoy1/train
```


2. For running orange_buoy files:
```
cd Code
python orange_buoy_1D_gauss.py 'video_path(in .mp4 format)' 'train_file_path'
```
where, video_path and train_file_path are the paths for input video and training data of the orange buoy. For example, running the python file on my local setup was:

```
cd Code/
python orange_buoy_1D_gauss.py /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/detectbuoy.avi /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/buoy2/train
```


```
cd Code
python orange_buoy_3D_gauss.py 'video_path(in .mp4 format)' 'train_file_path'
```
where, video_path and train_file_path are the paths for input video and training data of the orange buoy. For example, running the python file on my local setup was:

```
cd Code/
python orange_buoy_3D_gauss.py /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/detectbuoy.avi /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/buoy2/train
```


3. For running green_buoy files:
```
cd Code
python green_buoy_1D_gauss.py 'video_path(in .mp4 format)' 'train_file_path'
```
where, video_path and train_file_path are the paths for input video and training data of the green buoy. For example, running the python file on my local setup was:

```
cd Code/
python green_buoy_1D_gauss.py /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/detectbuoy.avi /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/buoy3/train
```


```
cd Code
python green_buoy_3D_gauss.py 'video_path(in .mp4 format)' 'train_file_path'
```
where, video_path and train_file_path are the paths for input video and training data of the green buoy. For example, running the python file on my local setup was:

```
cd Code/
python green_buoy_3D_gauss.py /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/detectbuoy.avi /home/arpitdec5/Desktop/color_segmentation_using_gmm_and_em/data/buoy3/train
```


### Software Required
To run the .py files, use Python 3. Standard Python 3 libraries like OpenCV, Numpy, matplotlib are used.
