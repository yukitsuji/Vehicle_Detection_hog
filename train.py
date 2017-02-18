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
from util import *

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
