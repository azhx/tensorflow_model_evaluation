<p align="left">
  <img src="https://cdn.discordapp.com/attachments/511941623299571713/546851305897852938/unknown.png" alt="Image of confusion_matrix.png" width=500 align= "center">
  <img src="https://cdn.discordapp.com/attachments/511941623299571713/546851514208223242/unknown.png" alt="Image of evaluation_metrics.csv" width=300 align = "center">
</p>

# Overview
This is a repo of two scripts used to generate evaluation metrics from models trained using Tensorflow's Object Detection API

This is based off an older version of the API which is the only version I got to work, but it should still work for newer versions if you have the files organized as below.

In reality, it will work as long as you have `object-detection.pbtxt` , `frozen_inference_graph.pb` , and a valid `Test.csv`

Submit an issue if there's a bug!!😀 

# Instructions
1. Check over the directory paths in `evaluate_test.py`<br />
2. run `python3 generate_evaluation_metrics.py`<br />
3. evaluate_test.py will make predictions on every image. This should take a bit of time, but there are logs. This will generate 2 important files `unifieddata.p` and `category_index.p`. If you already have these two files, you can change MODE in `generate_evaluation_metrics.py` to 2<br />
4. a file named `evaluation_metrics.csv` containing the evaluation metrics will be generated and a confusion matrix named `confusion_matrix.png` will be saved to the root directory<br />

`evaluation_metrics.csv` will contain 
`class | class accuracy | class f1score | class iou | overall accuracy | average of valid f1scores | average iou`

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

`utils` is from https://github.com/tensorflow/models/tree/master/research/object_detection<br /><br />
`frozen_inference_graph.pb` is the graph of whatever object detection model you've trained<br /><br />
`Test.csv` is a csv containing information about every image in the test set with the columns:<br /><br />
`filename | width | height | class | xmin | ymin | xmax | ymax`<br />

# Requirements

`numpy, tensorflow, pandas, pickle, matplotlib, PIL, os, sys`





