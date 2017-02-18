#!/user/bin/env python3
import time
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import glob
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC # for rbf
from sklearn.metrics import accuracy_score

def cal_hog_feature(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """Extract HOG feature from one color channel"""
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, feature_vector=feature_vec)
    return features

def extract_hog_features(image, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True, hog_color="RGB"):
    """Extract HOG features from colorspace you specify"""
    if hog_color != 'RGB':
        if hog_color == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif hog_color == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif hog_color == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif hog_color == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif hog_color == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)

    hog_features = []
    for channel in range(feature_image.shape[2]):
        hog_features.extend(cal_hog_feature(feature_image[:,:,channel],
                            orient, pix_per_cell, cell_per_block,
                            vis=viz, feature_vec=feature_vec))

def extract_features(images, hog_color='RGB', spatial_color="LUV", hist_color="HLS", spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """Extract features from HOG, color binning, and histogram of color space.
       You could choose features you wanna use.

       # Returns:
           features(ndarray): extracted features (1-dimensional array)
    """
    features = []
    for image in images:
        file_features = []
        if spatial_feat == True:
            spatial_features = bin_spatial(image, size=spatial_size, color=spatial_color)
            file_features.append(spatial_features)
        if hist_feat == True:
            hist_features = color_hist(image, nbins=hist_bins, color=hist_color)
            file_features.append(hist_features)
        if hog_feat == True:
            hog_features = extract_hog_features(image, orient, pix_per_cell, cell_per_block, hog_color=hog_color)
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    return np.array(features)

def bin_spatial(image, size=(32, 32), color="RGB"):
    """binning image of specified color space"""
    if color != 'RGB':
        if color == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)
    return cv2.resize(feature_image, size).ravel()

def color_hist(image, nbins=32, bins_range=(0, 256), color="RGB"):
    """convert RGB Image to specified color space histogram"""
    if color != 'RGB':
        if color == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)
    channel1_hist = np.histogram(feature_image[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(feature_image[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(feature_image[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features
