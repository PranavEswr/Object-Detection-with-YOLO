# Object-Detection-with-YOLO
In this project I have performed the object detection in images using YOLOv3 which is already pretrained on COCO-dataset under OpenCV and Python. Also applied non-maximum suppression to remove overlapping bounding boxes.

- YOLO (You Only Look Once) is a single-stage object detector, meaning it makes predictions for the bounding boxes and class labels of objects in an image in a single pass through the network.
- This makes it fast and efficient for real-time object detection applications but it may not be as accurate as some other object detection algorithms, such as R-CNN (Regional Convolutional Neural Network), which is a two-stage detector that uses a separate region proposal step followed by a CNN for classification.
- The key idea behind YOLO is to divide the input image into a grid of cells, with each cell responsible for predicting the bounding boxes and class probabilities for the objects present in that cell.
- Download YOLO v3 weights file from this [link](https://pjreddie.com/media/files/yolov3.weights)

  
# Variables
- confidence_thresh is used to filter out weak detections. In this case, we will use a value of 0.5, which means that only detections with a confidence score above 50% will be kept.
- NMS_thresh is used to eliminate overlapping bounding boxes. In our case, we will use a value of 0.3, which means that if the IoU (Intersection over Union) between two bounding boxes is above 30%, the bounding box with the lower confidence score will be discarded.
- green variable represents the green color that we will use to draw the bounding boxes on the input image.


# Why YOLO v3?
YOLO v3 is the third version of the YOLO object detection algorithm. It was introduced in 2018 as an improvement over YOLO v2, aiming to increase the accuracy and speed of the algorithm.
One of the main improvements in YOLO v3 is the use of a new CNN architecture called Darknet-53. Darknet-53 is a variant of the ResNet architecture and is designed specifically for object detection tasks. It has 53 convolutional layers and is able to achieve state-of-the-art results on various object detection benchmarks.
Another improvement in YOLO v3 are anchor boxes with different scales and aspect ratios. In YOLO v2, the anchor boxes were all the same size, which limited the ability of the algorithm to detect objects of different sizes and shapes. In YOLO v3 the anchor boxes are scaled, and aspect ratios are varied to better match the size and shape of the objects being detected.
YOLO v3 also introduces the concept of "feature pyramid networks" (FPN). FPNs are a CNN architecture used to detect objects at multiple scales. They construct a pyramid of feature maps, with each level of the pyramid being used to detect objects at a different scale. This helps to improve the detection performance on small objects, as the model is able to see the objects at multiple scales.
In addition to these improvements, YOLO v3 can handle a wider range of object sizes and aspect ratios. It is also more accurate and stable than the previous versions of YOLO.

