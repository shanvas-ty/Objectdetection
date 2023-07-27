# Objectdetection
A simple implementation of real-time object detection using the YOLOv3 model. It can be used to detect various objects in a webcam video stream.

                                    # Introduction
This is a Python program that performs real-time object detection using the YOLOv3 model. The program uses TensorFlow and OpenCV libraries to load the YOLOv3 model, 
process input images from a webcam, and draw bounding boxes around detected objects with their respective class labels.

        Requirements
        
TensorFlow
OpenCV
NumPy
wget (used for downloading YOLOv3 weights)

       Usage
Clone the repository or download the program file (e.g., object_detection_yolov3.py).

Download YOLOv3 pre-trained weights by running the weights_download() function inside the program. This will download the weights and save them in the specified location.

Prepare a text file (classes.TXT) containing the names of classes for the objects you want to detect, with each class name on a new line.

Run the program using a Python interpreter. Make sure your webcam is connected and accessible.

       Program Structure
Imports:
        The required libraries are imported, including TensorFlow, NumPy, and OpenCV.
YOLOv3 Model Definition:
                        Several helper functions and classes are defined to build the YOLOv3 model using TensorFlow.
                        Functions for DarknetConv, DarknetResidual, DarknetBlock, YoloConv, and YoloOutput are provided to create the YOLO model.
YOLOv3 Weights Loading:
                      The load_darknet_weights() function is used to load the pre-trained YOLOv3 weights into the custom model.
Webcam Video Capture:
                     The program captures video from the webcam using OpenCV.
Real-time Object Detection:
                          The video frames are processed in real-time using the YOLOv3 model.
                          The detected objects are drawn with bounding boxes and class labels on the video frames.

          Function Descriptions

load_darknet_weights(model, weights_file): Helper function to load pre-trained YOLOv3 weights into the custom model.

draw_outputs(img, outputs, class_names): Utility function to draw predicted bounding boxes and class labels on an image.

YoloV3(size=None, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks, classes=80): Function to create the YOLOv3 model.

yolo_boxes(pred, anchors, classes): Function to extract bounding boxes from YOLOv3 predictions.

yolo_nms(outputs, anchors, masks, classes): Function to perform Non-Maximum Suppression on YOLOv3 output.

weights_download(out='/home/shan/objectdetectioninspark/object/models/yolov3.weights'): Function to download YOLOv3 weights.
