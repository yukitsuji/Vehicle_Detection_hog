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


def augment_brightness_image(image, bright=1.25):
    """Apply brightness conversion to RGB image

       # Args
           image(ndarray): RGB image (3-dimension)
           bright(float) : bright value for multiple
       # Returns
           image(ndarray): RGB image (3-dimension)
    """
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1[:,:,2] = image1[:,:,2]*bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def make_datasets(car_parent_dir, noncar_parent_dir):
    """get pathes of car and non-car images from directories.
       # Args:
           car_parent_dir(str)   : car's parent directory
           noncar_parent_dir(str): non-car's parent directory

       # Returns:
           dataset_pathes(dic): dictionary of car and non-car path
                                key: ["car", "noncar"]
    """
    datasets_dir = {}
    car_list = np.array([])
    noncar_list = np.array([])
    car_directories = glob.glob(car_parent_dir)
    for car_dir in car_directories:
        car_list = np.hstack((car_list, np.array(glob.glob(car_dir + "/*.png"))))
        if car_dir == "./vehicles/KITTI_extracted":
            car_list = np.hstack((car_list, np.array(glob.glob(car_dir + "/*.png"))))
        if car_dir == "./vehicles/GTI_Right":
            car_list = np.hstack((car_list, np.array(glob.glob(car_dir + "/*.png"))))
        if car_dir == "./vehicles/GTI_Left":
            car_list = np.hstack((car_list, np.array(glob.glob(car_dir + "/*.png"))))

    noncar_directories = glob.glob(noncar_parent_dir)
    for noncar_dir in noncar_directories:
        noncar_list = np.hstack((noncar_list, np.array(glob.glob(noncar_dir + "/*.png"))))
        if noncar_dir == "./non-vehicles/Extras":
            noncar_list = np.hstack((noncar_list, np.array(glob.glob(noncar_dir + "/*.png"))))
            noncar_list = np.hstack((noncar_list, np.array(glob.glob(noncar_dir + "/*.png"))))

    datasets_dir = {"car" : car_list, "noncar" : noncar_list}
    return datasets_dir

def input_datasets(datasets, shape=(64, 64, 3)):
    """Input images from pathes of car and non-car.
       For adjust brightness, left-right balance, I apply data augmentation
       Apply Flip and Brightness conversion to GTI_Left and GTI_Right

       # Args:
           datasets(dic): pathes of datasets "car" and "non-car"
           shape(tuple) : shape of input images

       # Returns:
           input_images(ndarray): all images of datasets(4 dimension)
           Y (ndarray)          : all labels of datasets(1 (car) or 0 (non-car))
    """
    left_true = glob.glob("./vehicles/GTI_Left/*.png")
    len_left_true = len(left_true)
    right_true = glob.glob("./vehicles/GTI_Right/*.png")
    len_right_true = len(right_true)
    input_images = np.zeros((datasets["car"].shape[0] + datasets["noncar"].shape[0],
        shape[0], shape[1], shape[2]), dtype=np.uint8)
    input_images[:datasets["car"].shape[0]] = [io.imread(path) for path in datasets["car"]]
    input_images[datasets["car"].shape[0]:] = [io.imread(path) for path in datasets["noncar"]]

    augmented_images = np.zeros((len_left_true*2 + len_right_true*2,
        shape[0], shape[1], shape[2]), dtype=np.uint8)
    augmented_images[:len_left_true] = [augment_brightness_image(io.imread(path)) for path in left_true]
    augmented_images[len_left_true:len_left_true*2] = [cv2.flip(augment_brightness_image(io.imread(path), bright=1.1), 1) for path in left_true]
    augmented_images[2*len_left_true:2*len_left_true+len_right_true] = [augment_brightness_image(io.imread(path)) for path in right_true]
    augmented_images[2*len_left_true+len_right_true:] = [cv2.flip(augment_brightness_image(io.imread(path), bright=1.1), 1) for path in right_true]
    input_images = np.vstack((input_images, augmented_images))
    Y = np.hstack((np.ones((datasets["car"].shape[0])), np.zeros(datasets["noncar"].shape[0])))
    Y = np.hstack((Y, np.ones((len_left_true*2 + len_right_true*2))))
    return input_images, Y

def cal_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, feature_vector=feature_vec)
    return features

def extract_hog_features(image, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True, hog_color="RGB"):
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
        hog_features.extend(cal_hog_features(feature_image[:,:,channel],
                            orient, pix_per_cell, cell_per_block,
                            vis=viz, feature_vec=feature_vec))

def extract_features(images, hog_color='RGB', spatial_color="LUV", hist_color="HLS", spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
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

def main():
    datasets = make_datasets("./vehicles/*", "./non-vehicles/*")
    X, Y = input_datasets(datasets)
    X = extract_features(X, hog_color="YCrCb", hog_channel="ALL", hist_color="YCrCb", spatial_size=(16, 16),
        spatial_color="YCrCb", orient=18, cell_per_block=3, spatial_feat=True)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(
        scaled_X,
        Y,
        test_size=0.1,
        random_state=0,
    )
    svc = LinearSVC()
    # svc = SVC(kernel='rbf')
    t=time.time()
    svc.fit(X_train, Y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, Y_test), 4))
    t=time.time()
    n_predict = 1000
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', Y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    print(accuracy_score(Y_test, svc.predict(X_test)))

    print("saving svm parameter and scaler")
    np.savez("scaler.npz", mean=X_scaler.mean_, scale=X_scaler.scale_)
    with open("svm.pkl", mode="wb") as f:
        pickle.dump(svc, f)

if __name__ == "__main__":
    main()
