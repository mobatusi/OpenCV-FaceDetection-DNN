# Face Detection Using Dlib and DNN in OpenCV

This project demonstrates face detection using Dlib and OpenCV's Deep Neural Network (DNN) module. It showcases how to identify human faces in images, enabling applications like selfie enhancement, virtual avatar creation, and more.

## Features
- Detect faces in an image using DNN and Dlib.
- Label and visualize detected faces.
- Performance testing and comparison between DNN and Dlib methods.

## Project Workflow
1. **Import Libraries**: Set up the required dependencies.
2. **Download the Image**: Load a sample image for face detection.
3. **Load the DNN Network**: Initialize the DNN model for detection.
4. **Prepare Image and Run Network**: Process the image and detect faces using the DNN.
5. **Label and Visualize**: Draw bounding boxes and confidence scores around detected faces.
6. **Performance Test**: Evaluate the speed and accuracy of the DNN.
7. **Dlib Detection**: Prepare the image, run Dlib's detector, and visualize the results.
8. **Performance Comparison**: Compare DNN and Dlib's face detection performance.

## Prerequisites
- Python 3.x
- OpenCV
- Dlib
- NumPy
- Matplotlib

## Usage
1. Clone the repository:
   ```bash
   git https://github.com/mobatusi/OpenCV-FaceDetection-DNN.git
   cd OpenCV-FaceDetection-DNN
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook to execute the tasks.

## Results
- Visualize bounding boxes on faces detected using DNN and Dlib.
- Analyze performance metrics for each method.