import keras
from keras import backend as K

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

import tensorflow as tf
import pandas as pd
from os import listdir
from os.path import isfile, join
import tqdm

# for mscoco
# Images in input folder need to be named s.t they are pulled in time series order
def detect_images(func_detect_image, func_read_img, path_to_input_images, path_to_output_images=None, path_to_output_csv=None, output_PIL=True):

	labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

	image_filenames = [f for f in listdir(path_to_input_images) if isfile(join(path_to_input_images, f))]
	data = []
	image_filenames = sorted(image_filenames)

	for i, image_filename in tqdm(enumerate(image_filenames)):

		input_filename = os.path.join(path_to_input_images, image_filename)
		output_filename = os.path.join(path_to_output_images, 'processed_' + image_filename)
		
		orig_image = None
		try:
			orig_image, image = func_read_img(input_filename)
		except:
			print('{} is not a valid image file'.format(image_filename))
			continue

		plt, detection = func_detect_image(image, path_to_output_images, orig_image)
		data.append(detection)
		
		plt.savefig(output_filename)

	output_csv = os.path.join(path_to_output_csv, 'results.csv')
	df = pd.DataFrame(np.array(data))
	df.to_csv(output_csv, header=list(labels_to_names.values()), index=False)


