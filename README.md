# U-Net for segmentation
This is the reference implementation of the models and code for the U-Net which was proposed by O Ronneberger,P Fischer and T Brox.
You can set parameters like file directory, image size, number of classes etc. in configuration.txt, then run U-Net_Training.py to train the model, run U-Net_Predict.py to segment images.
It's mainly for gray images. In my own seg task, the gt mask is saved as png(4 classes and 1 for bg), and the corresponding gray value is [0, 85, 170, 255]
