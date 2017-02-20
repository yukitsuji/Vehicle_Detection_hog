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


def make_datasets(car_parent_dir, noncar_parent_dir):
    datasets_dir = {}
    car_list = np.array([])
    noncar_list = np.array([])
    car_directories = glob.glob(car_parent_dir)
    for car_dir in car_directories:
        car_list = np.hstack((car_list, np.array(glob.glob(car_dir + "/*.png"))))

    noncar_directories = glob.glob(noncar_parent_dir)
    for noncar_dir in noncar_directories:
        noncar_list = np.hstack((noncar_list, np.array(glob.glob(noncar_dir + "/*.png"))))

    datasets_dir = {"car" : car_list, "noncar" : noncar_list}
    return datasets_dir

def input_datasets(datasets, shape=(64, 64, 3)):
    kitti_true = glob.glob("./vehicles/KITTI_extracted/*.png")
    len_kitti_true = len(kitti_true)
    kitti_false = glob.glob("./non-vehicles/Extras/*.png")
    len_kitti_false = len(kitti_false)
    input_images = np.zeros((datasets["car"].shape[0] + datasets["noncar"].shape[0] + len_kitti_false + len_kitti_true,
                             shape[0], shape[1], shape[2]), dtype=np.uint8)
    input_images[:datasets["car"].shape[0]] = [io.imread(path) for path in datasets["car"]]
    input_images[datasets["car"].shape[0]:datasets["car"].shape[0] + len_kitti_true]= [cv2.flip(io.imread(path), 1) for path in kitti_true]
    input_images[datasets["car"].shape[0] + len_kitti_true : datasets["car"].shape[0] + len_kitti_true + datasets["noncar"].shape[0]] = \
        [io.imread(path) for path in datasets["noncar"]]
    input_images[datasets["car"].shape[0] + len_kitti_true + datasets["noncar"].shape[0]:] = \
        [cv2.flip(io.imread(path), 1) for path in kitti_false]

    Y = np.hstack((np.ones((datasets["car"].shape[0] + len_kitti_true)), np.zeros(datasets["noncar"].shape[0] + len_kitti_false)))
    return input_images, Y

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, feature_vector=feature_vec)
    return features

def extract_feature(image, hog_color='RGB', spatial_color="LUV", hist_color="HLS", spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    file_features = []
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

    if spatial_feat == True:
        spatial_features = bin_spatial(image, size=spatial_size, color=spatial_color)
        file_features.append(spatial_features)
    if hist_feat == True:
        hist_features = color_hist(image, nbins=hist_bins, color=hist_color)
        file_features.append(hist_features)
    if hog_feat == True:
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=True))
        file_features.append(hog_features)
    return np.concatenate(file_features)

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

def overlapping_area(detection_1, detection_2):
    '''
    Function to calculate overlapping area'si
    `detection_1` and `detection_2` are 2 detections whose area
    of overlap needs to be found out.
    Each detection is list in the format ->
    [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
    The function returns a value between 0 and 1,
    which represents the area of overlap.
    0 is no overlap and 1 is complete overlap.
    '''
    x1_tl = detection_1[0][0]
    x2_tl = detection_2[0][0]
    x1_br = detection_1[1][0]
    x2_br = detection_2[1][0]
    y1_tl = detection_1[0][1]
    y2_tl = detection_2[0][1]
    y1_br = detection_1[1][0]
    y2_br = detection_2[1][1]
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    # area_1 = detection_1[3] * detection_2[4]
    # area_2 = detection_2[3] * detection_2[4]
    area_1 = (x1_br - x1_tl) * (y2_br - y2_tl)
    area_2 = (x2_br - x2_tl) * (y2_br - y2_tl)
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

def nms(detections, threshold=.5):
    """
    This function performs Non-Maxima Suppression.
    `detections` consists of a list of detections.
    Each detection is in the format ->
    [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
    If the area of overlap is greater than the `threshold`,
    the area with the lower confidence score is removed.
    The output is a list of detections.
    """
    print("detections", detections)
    if len(detections) == 0:
	       return []
    detections = sorted(detections, key=lambda detections: detections[1],
            reverse=True)
    new_detections=[]
    new_detections.append(detections[0][0])
    del detections[0]
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if overlapping_area(detection[0], new_detection) > threshold:
                del detections[index]
                break
        else:
            new_detections.append(detection[0])
            del detections[index]
    return new_detections

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

    # xy_window=(200, 140)#(84, 64), (120, 96), (150, 110), (200, 140)
    window_list = []
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    rate = 1/2
    xy_window_list = [[200, 170, int(yspan*rate), yspan, 0, xspan],
                      [150, 120, int(yspan*rate*6 / 4), yspan-int(yspan*rate*rate*rate), 0, xspan],
                      [120, 110, int(yspan*rate*6 / 4), yspan-int(yspan*rate*rate), 0, xspan],
                      [120, 96, int(yspan*rate*rate*rate), yspan-int(yspan*rate), 0, xspan],
                      [84, 64, 0, int(yspan*rate*rate), 0, xspan]]
    # xy_window_list = [[200, 170, int(yspan*rate), yspan, 0, xspan],
    #                   [150, 120, int(yspan*rate*6 / 4), yspan-int(yspan*rate*rate*rate), 0, xspan],
    #                   [120, 110, int(yspan*rate*6 / 4), yspan-int(yspan*rate*rate), 0, xspan],
    #                   [120, 96, int(yspan*rate*rate*rate), yspan-int(yspan*rate), 0, xspan],
    #                   [84, 64, 0, int(yspan*rate*rate), 0, xspan]]
                    #   [150, 120, int(yspan*rate*6 / 4), yspan-int(yspan*rate*rate), 0, xspan] before images10
                    # [150, 120, int(yspan*rate*6 / 4), yspan-int(yspan*rate*rate*rate), 0, xspan] after images10 and delete images9
                    #   [120, 96, int(yspan*rate*rate*rate), yspan-int(yspan*rate), 0, xspan] before images9
                    # [120, 96, int(yspan*rate*rate*rate), yspan-int(yspan*rate*rate), 0, xspan] after images9
                    #   [84, 64, 0, int(yspan*rate*rate), int(xspan / 6), int(xspan / 6 * 5)] images7?

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
        features = extract_feature(test_img, hog_color=hog_color, hog_channel=hog_channel, spatial_feat=spatial_feat,
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
    scaler = np.load("final2.npz")
    X_scaler = StandardScaler()
    X_scaler.mean_, X_scaler.scale_ = scaler["mean"], scaler["scale"]
    with open("final2.pkl", mode="rb") as f:
        svc = pickle.load(f)

    cap = cv2.VideoCapture("project_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (1280,720))
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
