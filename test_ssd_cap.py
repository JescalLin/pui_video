import numpy as np
import argparse
import tensorflow as tf
import cv2
import pathlib

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,  480))


def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    output_dict = model(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                    output_dict['detection_masks'], output_dict['detection_boxes'],
                                    image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict



def box_area_check(img, boxes, scores):
    box_loc = []
    height = img.shape[0]
    width = img.shape[1]
    for i in range(len(scores)):
        if scores[i]< 0.5:
            scores[i] = 0
            boxes[i] = 0

        x=int(boxes[i][1]*width)
        y=int(boxes[i][0]*height)
        w=int(boxes[i][3]*width)-int(boxes[i][1]*width)
        h=int(boxes[i][2]*height)-int(boxes[i][0]*height)

        if w!=0:
            box_loc.append([x,y,w,h])
    return box_loc

def run_inference(model, category_index, cap):
    fps = 0
    while True:
        ret, image_np = cap.read()
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        boxes = output_dict['detection_boxes']
        scores = output_dict['detection_scores']

        box_list = box_area_check(image_np,boxes,scores)

        for box in box_list:
            x,y,w,h = box
            cv2.rectangle(image_np, (x,y), (x+w,y+h), (255,0,0), 5)

        
        if len(box_list)==0:
            fps = fps+1
        else:
            fps = 0

        if fps == 15:
            out.write(image_np)
            fps = 0
        print(fps)
        cv2.imshow('object_detection', cv2.resize(image_np, (640, 480)))
        key = cv2.waitKey(90)
        if key == 13:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':

    # model_name = "box_frcnn800_20210127"
    # PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
    # detection_model = load_model(model_name)
    # category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


    # model_name = "box_graph_rcnn_resnet101_all"
    model_name = "hand_inference_graph"
    PATH_TO_LABELS = 'hand.pbtxt'
    detection_model = tf.saved_model.load(str(model_name+"/saved_model"))
    detection_model = detection_model.signatures['serving_default']
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


    cap = cv2.VideoCapture(0)
    run_inference(detection_model, category_index, cap)