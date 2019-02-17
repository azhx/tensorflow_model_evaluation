import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

sys.path.append("..")

#EDIT BELOW
#########################################################################
# What model to download. change as needed
MODEL_NAME = 'unaugmented1'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

#only scores above this threshold will be taken as predictions.
min_score_thresh = 0.5
##########################################################################

NUM_CLASSES = 90 # this is the default value- doesn't need to be changed
#Load graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#Load Label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


#Read Testcsv from current directory
testcsv = pd.read_csv('test.csv')
testcsv = testcsv.sort_values(by=['filename'])
testcsv = testcsv.reset_index(drop=True)

#generate prediciton dict
predict = {'filename':[],
           'npreds': [],
           'class': [],
           'score': [],
           'xmin': [],
           'ymin': [],
           'xmax': [],
           'ymax': []}

#prediction Session
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for k, filename in enumerate(images):
            image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, filename)
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            print ('predicting', k)
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            #getting data from prediction
            xmins, ymins, xmaxs, ymaxs, classlist = [], [], [], [], []
            npreds = len(scores[scores>min_score_thresh])
            classes = np.squeeze(classes)[:npreds].astype(np.int32)
            boxes = np.squeeze(boxes)[:npreds]
            for i in range(npreds):
                classlist.append(category_index[classes[i]]['name'])
                xmins.append(boxes[i][1]) #outputs from sess.run return x,y values in different order than test.csv
                ymins.append(boxes[i][0])
                xmaxs.append(boxes[i][3])
                ymaxs.append(boxes[i][2])
            predict['filename'].append(filename)
            predict['npreds'].append(npreds)
            predict['class'].append(classes)
            predict['score'].append(np.squeeze(scores)[:npreds])
            predict['xmin'].append(xmins)
            predict['ymin'].append(ymins)
            predict['xmax'].append(xmaxs)
            predict['ymax'].append(ymaxs)

#writes to a csv
pd.DataFrame(data = predict).to_csv('predict.csv')

#writes dict to a pickle file
predict_byimage = {}
for k, v in enumerate(predict['filename']):
    predict_byimage[v] = {'npreds': predict['npreds'][k],
           'class': predict['class'][k],
           'score': predict['score'][k],
           'xmin': predict['xmin'][k],
           'ymin': predict['ymin'][k],
           'xmax': predict['xmax'][k],
           'ymax': predict['ymax'][k]}
pickle.dump(predict_byimage, open("predictlookup.p", "wb"))

#writes sorted test csv
testcsv.to_csv('sorted_test.csv')
