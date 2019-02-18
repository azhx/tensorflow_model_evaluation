# tensorflow_model_evaluation

# Overview
Scripts to generate evaluation metrics from models trained using Tensorflow's Object Detection API

This is based off an older version of the API which is the only version I got to work, but it should still work for newer versions if you have the files organized like this.

# Directory Tree
```
.
├── training
│   └──model_ckpt_folder
│      └── object-detection.pbtxt
├── utils
│   ├── label_map_util.py
│   └── visualization_utils.py
├── model_name_folder
│   └── frozen_inference_graph.pb
├── Test
│   ├──1.jpg
│   ├──2.jpg
│   ├──3.jpg
│   └──etc.jpg
├── Test.csv
├── evaluate_test.py
└── generate_evaluation_metrics.py

```
<br />
`utils` is from https://github.com/tensorflow/models/tree/master/research/object_detection<br />
`frozen_inference_graph.pb` is the graph of whatever object detection model you've trained<br />
`Test.csv` is a csv containing information about every image in the test set with the columns:<br />
`filename | width | height | class | xmin | ymin | xmax | ymax`<br />

## Instructions
1. Check over the directory paths in `evaluate_test.py`<br />
2. run `python3 generate_evaluation_metrics.py`<br />
3. evaluate_test.py will make predictions on every image. This should take a bit of time, but there are logs. This will generate 2 important files `unifieddata.p` and `category_index.p`. If you already have these two files, you can change MODE in `generate_evaluation_metrics.py` to 2<br />
4. a file named `evaluation_metrics.csv` containing the evaluation metrics will be generated.<br />

