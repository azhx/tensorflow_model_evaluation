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

"""
Global variables to be set
"""

# Relative path to the model folder containing your graph
MODEL_NAME = 'unaugmented1'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
#only scores above this threshold will be taken as predictions.
min_score_thresh = 0.5
#Directory containing images of Test set
PATH_TO_TEST_IMAGES_DIR = 'Test'
# this is the default value- doesn't need to be changed
NUM_CLASSES = 90


def _run_test_set_prediction():
    """
    Code from older version of the tensorflow object detection api example:
    https://github.com/tensorflow/models/blob/3543f02d214cdd9798b4603fa8783652d222a73f/research/object_detection/object_detection_tutorial.ipynb
    """
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

    #category_index will be needed later
    pickle.dump(category_index, open('category_index.p', 'wb'))

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

    #generate list of images to iterate through
    images = os.listdir(PATH_TO_TEST_IMAGES_DIR)

    #prediction Session
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            print ('tensorflow session launched')
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
                print (f'{k}/{len(images)}', ': ', filename)
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
                predict['class'].append(list(classes))
                predict['score'].append(list(np.squeeze(scores)[:npreds]))
                predict['xmin'].append(xmins)
                predict['ymin'].append(ymins)
                predict['xmax'].append(xmaxs)
                predict['ymax'].append(ymaxs)
    return predict, testcsv, category_index


def _output_and_write_data(predict, testcsv, category_index):
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

    #These is mainly for debugging purposes
    pickle.dump(predict, open("predict.p", "wb"))
    #writes sorted test csv
    testcsv.to_csv('sorted_test.csv')
    #category_index required in other scripts
    pickle.dump(category_index, open("category_index.p", "wb"))
    return


def _generate_truedict(testcsv):
    truedict= {}
    for each in testcsv['filename']:
        truedict[each] = {
            'npreds': 0,
            'class': [],
            'xmin':[],
            'ymin':[],
            'xmax':[],
            'ymax':[]
        }
    for i, each in enumerate(testcsv['filename']):
        truedict[each]['npreds'] += 1
        truedict[each]['class'].append(testcsv.at[i, 'class'])
        truedict[each]['xmin'].append(testcsv.at[i, 'xmin']/testcsv.at[i, 'width'])
        truedict[each]['ymin'].append(testcsv.at[i, 'ymin']/testcsv.at[i, 'height'])
        truedict[each]['xmax'].append(testcsv.at[i, 'xmax']/testcsv.at[i, 'width'])
        truedict[each]['ymax'].append(testcsv.at[i, 'ymax']/testcsv.at[i, 'height'])

    return truedict


def _generate_unified_data(predict, truedict, category_index):
    unified_data = predict
    unified_data['truenpreds'] = 0
    unified_data['trueclass'] = ''
    unified_data['truexmin'] = None
    unified_data['trueymin'] = None
    unified_data['truexmax'] = None
    unified_data['trueymax'] = None

    for i, each in enumerate(unified_data['filename']):
        if unified_data.at[i, 'class'] == []:
            unified_data.at[i, 'class'] = None
        else:
            if (type(unified_data.at[i, 'class']) == str):
                unified_data.at[i, 'class'] = eval(unified_data.at[i, 'class'])
            for j in range(len(unified_data.at[i, 'class'])):
                unified_data.at[i, 'class'][j] = category_index[unified_data.at[i, 'class'][j]]['name']
        unified_data.at[i, 'truenpreds'] = truedict[each]['npreds']
        unified_data.at[i, 'trueclass'] = truedict[each]['class']
        unified_data.at[i, 'truexmin'] = truedict[each]['xmin']
        unified_data.at[i, 'trueymin'] = truedict[each]['ymin']
        unified_data.at[i, 'truexmax'] = truedict[each]['xmax']
        unified_data.at[i, 'trueymax'] = truedict[each]['ymax']
        unified_data.at[i, 'xmin'] = eval(unified_data.at[i, 'xmin'])
        unified_data.at[i, 'ymin'] = eval(unified_data.at[i, 'ymin'])
        unified_data.at[i, 'xmax'] = eval(unified_data.at[i, 'xmax'])
        unified_data.at[i, 'ymax'] = eval(unified_data.at[i, 'ymax'])

    #WRITES FILE
    pickle.dump(unified_data, open('unifieddata.p', 'wb'))
    return

def evaluate_test_set(MODE = 'RUN'):

    if MODE == 'RUN':
        predict, testcsv, category_index = _run_test_set_prediction()
        _output_and_write_data(predict, testcsv, category_index)
        truedict = _generate_truedict(testcsv)
        _generate_unified_data(predict, truedict, category_index)

    if MODE == 'DEBUG':
        predict = pd.read_csv('predict.csv')
        testcsv = pd.read_csv('test.csv')
        testcsv = testcsv.sort_values(by=['filename'])
        testcsv = testcsv.reset_index(drop=True)
        category_index = pickle.load(open("category_index.p", "rb"))
        _output_and_write_data(predict, testcsv, category_index)
        truedict = _generate_truedict(testcsv)
        _generate_unified_data(predict, truedict, category_index)
        print ('successful execution')
    return
