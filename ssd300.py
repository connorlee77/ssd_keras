from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from eval_utils.coco_utils import get_coco_category_maps, predict_all_to_json

import results

# Set the input image size for the model.
img_height = 300
img_width = 300

K.clear_session() # Clear previous models from memory.
model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=80,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], # The scales for Pascal VOC are [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

weights_path = 'weights/VGG_coco_SSD_300x300_iter_400000.h5'
model.load_weights(weights_path, by_name=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

MS_COCO_dataset_annotations_filename = 'instances_val2017.json'
cats_to_classes, classes_to_cats, cats_to_names, classes_to_names = get_coco_category_maps(MS_COCO_dataset_annotations_filename)


def read_image(filepath):
    img_height = 300
    img_width = 300

    img = image.load_img(filepath, target_size=(img_height, img_width))

    img = image.img_to_array(img)
    input_image = np.array([img])
    return [imread(filepath)], input_image

def detect_image(input_images, output_image_path, orig_image):
    img_height = 300
    img_width = 300

    # ## 3. Make predictions
    y_pred = model.predict(input_images)
    confidence_threshold = 0.5

    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)

    colors = plt.cm.hsv(np.linspace(0, 1, 80)).tolist()

    plt.figure()
    plt.imshow(orig_image[0])

    current_axis = plt.gca()

    detections = np.zeros(80)
    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_image[0].shape[1] / img_width
        ymin = box[3] * orig_image[0].shape[0] / img_height
        xmax = box[4] * orig_image[0].shape[1] / img_width
        ymax = box[5] * orig_image[0].shape[0] / img_height

        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes_to_names[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

        detections[int(box[0]) - 1] += 1

    return plt, detections

path_to_input_images = 'input'
path_to_output_images = 'output'
path_to_output_csv = 'csv'

results.detect_images(detect_image, read_image, path_to_input_images, path_to_output_images, path_to_output_csv, output_PIL=False)



