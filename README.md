# Computer_Vision_Using_TensorFlowLite

On this project the AlexNet Convolution Neural Network is trained using traffic sign images from the German Road Traffic Sign Benchmark.
The initially trained network is then quantized/optimized for deployment on mobile devices using TensorFlow Lite 

## Project key Steps

1. Download German Traffic Sign Benchmark training images [from here](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip)
2. Convert from .ppm to .jpg format
3. Label db folder appropriately
4. Convert dataset to TFRecord
5. Create CNN (alexnet)
6. Train CNN using TFRecord data
7. Test & evaluate trained model
8. Quantize/Tune/Optimize trained model for mobile deployment
9. Test & evaluate tuned model



