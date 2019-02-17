# tensorflow_model_evaluation
Scripts to generate evaluation metrics from models trained using Tensorflow's Object Detection API

```
.
├── training
│   └──model_ckpt_folder_name
│      └── object-detection.pbtxt
├── utils
│   ├── label_map_util.py
│   └── visualization_utils.py
├── model_ckpt_folder
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
