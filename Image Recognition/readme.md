A complex convolution neural network (CNN) to recognize images. The dataset used here is the CIFAR-10 dataset (http://www.cs.toronto.edu/~kriz/ cifar.html). The objective of the network is to take RGB images and successfully recognize and classify them into 10 classes.

How to Run:
</br>1. Install Tensor Flow by following instructions from
https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html
</br>2. Enter the following
</br>source activate tensorflow
</br>Python cifar10_train.py
</br>python cifar10_eval.py
</br>tensorboard --logdir=/tmp/cifar10_train
</br>3. Go to http://localhost:6006/ to view graphs and networks


Details: 

• Convolution layer 1 (Kernel: 5x5x3, Stride: 1, Features: 64 )  
• Max Pooling layer 1 (Stride 2x2)  
• Local Response Normalization  
• Convolution layer 2 (Kernel: 5x5x64, Stride: 1, Features: 64 )  
• Max Pooling layer 2 (Stride 2x2)  
• Local Response Normalization  
• Fully connected Layer 1(Neurons : 384)  
• Fully connected layer 2(Neurons : 192)  
• Softmax Layer  

Accuracy:  
To test the accuracy we run cifar10_eval.py  
Parameters to train : 1,068,298  
Reached 82.1% accuracy after running for 4 hours on an Intel i7 CPU Reaches 86% accuracy after running overnight  
  
Code Info:  
Cifar10.py – Builds the entire network model  
Cifar10_train.py – trains the model  
Cifar10_eval.py – Evaluates the performance of the trained network  
  
Input:  
Images are cropped to 24x24  
Distort image brightness and contrast Randomly flip images  

