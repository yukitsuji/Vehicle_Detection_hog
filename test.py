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
from scipy.ndimage.measurements import label
from util import *

def slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = image.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = image.shape[0]

    window_list = []
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    rate = 1/2
    xy_window_list = [[200, 170, int(yspan*rate), yspan, 0, xspan],
                      [150, 120, int(yspan*rate*6 / 4), yspan-int(yspan*rate*rate*rate), 0, xspan],
                      [120, 110, int(yspan*rate*6 / 4), yspan-int(yspan*rate*rate), 0, xspan],
                      [120, 96, int(yspan*rate*rate*rate), yspan-int(yspan*rate), 0, xspan],
                      [84, 64, 0, int(yspan*rate*rate), 0, xspan]]

    for xy_window in xy_window_list:
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

        nx_windows = np.int((xy_window[5] - xy_window[4]) / nx_pix_per_step) - 1
        ny_windows = np.int((xy_window[3] - xy_window[2]) / ny_pix_per_step) - 1

        for ys in range(ny_windows):
            for xs in range(nx_windows):
                startx = xs * nx_pix_per_step + x_start_stop[0] + xy_window[4]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0] + xy_window[2]
                endy = starty + xy_window[1]
                window_list.append(((startx, starty), (endx, endy)))
    return window_list

def draw_boxes(image, bboxes, color=(0, 0, 255), thick=6):
    new_image = image.copy()
    for bbox in bboxes:
        cv2.rectangle(new_image, bbox[0], bbox[1], color, thick)
    return new_image

def add_heat(heatmap, bbox_list, value=1):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += value
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def search_windows(image, windows, svc, scaler, hog_color="RGB", hog_channel="ALL",
                   spatial_size=(32, 32), spatial_color="RGB", hist_color="HLS", hist_feat=True,
                   spatial_feat=True, orient=9, cell_per_block=3):
    on_windows = []
    for window in windows:
        test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = extract_features(test_img, hog_color=hog_color, hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    spatial_size=spatial_size, spatial_color=spatial_color, hist_color=hist_color,
                                    hist_feat=hist_feat, orient=orient, cell_per_block=cell_per_block)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = svc.predict(test_features)
        if prediction == 1:
            if svc.decision_function(test_features)[0] > 0.3:
                on_windows.append([window, svc.decision_function(test_features)[0]])
    return on_windows

def draw_labeled_bboxes(img, labels):
    bbox_list = []
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_list.append(bbox)
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    return img, bbox_list

def main():
    scaler = np.load("final.npz")
    X_scaler = StandardScaler()
    X_scaler.mean_, X_scaler.scale_ = scaler["mean"], scaler["scale"]
    with open("final.pkl", mode="rb") as f:
        svc = pickle.load(f)

    cap = cv2.VideoCapture("project_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280,720))
    heatmap = None
    ex_bbox_list = []
    exex_bbox_list = []
    exexex_bbox_list = []
    exexexex_bbox_list = []
    index=0
    while(1):
        ret, image = cap.read()
        index+=1
        heatmap = np.zeros_like(image)
        print("index", index)
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        windows = slide_window(converted_image, x_start_stop=[None, None], y_start_stop=[400, image.shape[0]-200],
                    xy_window=(240, 160), xy_overlap=(0.9, 0.9))
        bboxes = search_windows(converted_image, windows, svc, X_scaler, hog_color="YCrCb", hog_channel="ALL",
                                spatial_size=(16, 16), spatial_color="YCrCb", hist_color="YCrCb",
                                spatial_feat=True, orient=18, cell_per_block=2)
        if bboxes:
            heatmap = add_heat(heatmap, np.array(bboxes)[:, 0], value=2)
        if ex_bbox_list:
            heatmap = add_heat(heatmap, ex_bbox_list, value=2)
        if exex_bbox_list:
            heatmap = add_heat(heatmap, exex_bbox_list, value=2)
        if exexex_bbox_list:
            heatmap = add_heat(heatmap, exexex_bbox_list, value=1)
        if exexexex_bbox_list:
            heatmap = add_heat(heatmap, exexexex_bbox_list, value=1)
        heatmap = apply_threshold(heatmap, threshold=6)
        labels = label(heatmap)
        exexexex_bbox_list = exexex_bbox_list
        exexex_bbox_list = exex_bbox_list
        exex_bbox_list = ex_bbox_list
        window_img, ex_bbox_list = draw_labeled_bboxes(image, labels)
        out.write(window_img)
        # window_img = draw_boxes(image, bboxes, color=(0, 0, 255), thick=6)
        # cv2.imwrite("./images13/detected" + str(index) + ".jpg", window_img)
    out.release()
    cap.release()

if __name__ == "__main__":
    main()
