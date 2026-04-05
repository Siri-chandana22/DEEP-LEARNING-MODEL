# DEEP-LEARNING-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: YARLAGADDA SIRI CHANDANA

*INTERN ID*: CT4MDR3097

*DOMAIN*: DATA SCIENCE

*DURATION*: 4 MONTHS

*MENTOR*: NEELA SANTOSH

# Deep Learning Image Classification using Fashion MNIST

It contains a Deep Learning–based image classification project implemented using TensorFlow and Keras. The goal of this project is to classify grayscale fashion images into different clothing categories using a Convolutional Neural Network (CNN). The project demonstrates the complete deep learning workflow including dataset loading, preprocessing, visualization, model building, training, evaluation, and prediction. The Fashion MNIST dataset is used in this project, which is a popular benchmark dataset for image classification tasks. It consists of 70,000 grayscale images of clothing items divided into 10 classes, making it ideal for demonstrating convolutional neural network-based classification.

## Overview of the Implementation

The implementation is structured in multiple steps, beginning with dataset loading and ending with prediction visualization. Each step is clearly organized to ensure readability and modularity, allowing users to understand the complete deep learning pipeline.

### 1. Loading the Dataset

The project starts by loading the Fashion MNIST dataset using TensorFlow’s built-in dataset loader. The dataset contains 60,000 training images and 10,000 testing images. Each image is 28x28 pixels in grayscale format. The dataset includes ten categories of clothing items such as T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. After loading, class labels are mapped to human-readable names, which are later used for visualization and prediction display.

### 2. Data Normalization

Before training the model, preprocessing is applied to the dataset. The pixel values range from 0 to 255, so they are normalized by dividing by 255.0. This converts the values into a range between 0 and 1. Normalization improves training stability and helps the neural network converge faster. It also ensures consistent input scaling across all images.

### 3. Adding Channel Dimension

Since convolutional neural networks require a channel dimension, the dataset is reshaped to include one channel. The original dataset has shape (28, 28), which is converted to (28, 28, 1). This step ensures compatibility with Conv2D layers in the CNN architecture. Because the images are grayscale, only a single channel is used.

### 4. Visualizing Sample Images

To verify that the dataset is loaded correctly, sample images are displayed using Matplotlib. The code plots multiple images along with their respective labels. This visualization step helps confirm correct preprocessing and provides a clear understanding of the dataset before training the model.

### 5. Building the CNN Model

The classification model is built using a Convolutional Neural Network. The architecture consists of convolutional layers followed by max pooling layers. The convolutional layers extract important spatial features such as edges, patterns, and textures from the images. Max pooling layers reduce the spatial dimensions and improve computational efficiency.

After feature extraction, the output is flattened and passed to fully connected dense layers. The dense layer learns complex feature relationships. The final output layer contains 10 neurons corresponding to the ten clothing categories. A softmax activation function is used in the output layer to produce probability scores for each class.

### 6. Model Compilation

The model is compiled using the Adam optimizer, sparse categorical crossentropy loss function, and accuracy as the evaluation metric. The Adam optimizer provides adaptive learning rates, which improves training performance. Sparse categorical crossentropy is used because labels are integer encoded.

### 7. Model Training

The model is trained using the training dataset for multiple epochs. During training, the model learns feature representations and updates weights using backpropagation. Validation is performed using the testing dataset to measure performance on unseen data. Training and validation accuracy and loss values are stored for later visualization.

### 8. Accuracy and Loss Visualization

After training, the model performance is visualized using accuracy and loss graphs. The accuracy graph shows how well the model improves during training. The loss graph shows how prediction error decreases over time. These graphs help evaluate whether the model is learning correctly and detect potential overfitting.

### 9. Predictions and Result Visualization

Once training is complete, the model generates predictions on the test dataset. The predicted labels are compared with actual labels. Sample test images are displayed along with predicted and actual class names. This step helps visually verify classification performance and demonstrates the effectiveness of the trained CNN model.

## Key Features

### Complete Deep Learning Workflow

This project demonstrates the entire deep learning pipeline from dataset loading to prediction visualization.

### CNN-Based Image Classification

Uses convolutional neural networks for effective feature extraction and classification.

### Data Normalization

Improves model training stability and convergence speed.

### Prediction Visualization

Displays predicted and actual labels for easy comparison.

### Performance Graphs

Accuracy and loss plots help evaluate model learning behavior.

## Conclusion

This project demonstrates a complete deep learning-based image classification system using the Fashion MNIST dataset. It integrates data preprocessing, CNN model building, training, evaluation, and prediction into a single workflow. The pipeline provides a strong foundation for understanding image classification using convolutional neural networks.

## Results

<img width="1121" height="449" alt="Image" src="https://github.com/user-attachments/assets/9468b7ce-749d-450e-a15f-74b393258981" />

<img width="1281" height="737" alt="Image" src="https://github.com/user-attachments/assets/955ce459-10bf-4f76-9365-abba63272166" />

<img width="963" height="532" alt="Image" src="https://github.com/user-attachments/assets/009f2f39-3299-4ab8-a5fc-1549b98595d8" />

<img width="1005" height="549" alt="Image" src="https://github.com/user-attachments/assets/7c37f1a8-ab3f-440c-a038-c14e7a86571a" />
