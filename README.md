# YOLO in Image and Video

To apply real-time object detection, we are using deep learning and OpenCV to work with image and video files.</br>
The algorithm applies a neural network to an entire image. The network divides the image into an S x S grid and comes up with bounding boxes, which are boxes drawn around images and predicted probabilities for each of these regions.</br>

## Setup
1. Install OpenCV 3.3.0 or higher version on your system. 
```
For Ubuntu, you can refer https://www.learnopencv.com/install-opencv3-on-ubuntu/
For Mac, you can refer https://www.learnopencv.com/install-opencv3-on-macos/
```
2. Create a Python 3 virtual environment for OpenCV called cv:</br>
```mkvirtualenv cv -p python3```

3. pip install OpenCV into your new environment:</br>
```pip install opencv-contrib-python```

## Demo
* For Images,</br>
python yolo.py --image images/XYZ.jpg --yolo yolo-coco</br>
<img src="https://user-images.githubusercontent.com/26343062/67140242-9af8bd00-f20d-11e9-9c53-5ac166de2325.jpeg" width="350" height="350">

* For Videos,</br>
python yolo_video.py --input videos/XYZ.mp4 \--output output/XYZ.avi --yolo yolo-coco
<video src="airplane.mp4" width="320" height="200" controls preload></video>
