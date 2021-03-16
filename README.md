# AutoEncoder-Reconstruction
autoencoder for mnist dataset reconstruction

## DISCRIPTION
> Autoencoders are a type of unsupervised neural network (i.e., no class labels or labeled data) that seek to:
>
> * 1. Accept an input set of data (i.e., the input).
> 
> * 2. Internally compress the input data into a latent-space representation (i.e., a single vector that compresses and quantifies the input).
>
> * 3. Reconstruct the input data from this latent representation (i.e., the output).
> 
> Typically, we think of an autoencoder having two components/subnetworks:
> 
> * 1. Encoder: Accepts the input data and compresses it into the latent-space. If we denote our input data as x and the encoder as E, then the output latent-space representation, s, would be s = E(x).
> 
> * 2. Decoder: The decoder is responsible for accepting the latent-space representation s and then reconstructing the original input. If we denote the decoder function as D and the output of the detector as o, then we can represent the decoder as o = D(s).
> 
> Using our mathematical notation, the entire training process of the autoencoder can be written as:
> 
> ![image](https://user-images.githubusercontent.com/53394692/111267391-18ae3b80-8641-11eb-84ea-6f39cca15b1e.png)
>
> below demonstrates the basic architecture of an autoencoder:
>
> ![image](https://user-images.githubusercontent.com/53394692/111267092-b9502b80-8640-11eb-92b1-89b1f8001c5e.png)

## DATASET  
> Later in this tutorial, we’ll be training an autoencoder on the MNIST dataset. The MNIST dataset consists of digits that are 28×28 pixels with a single channel, implying that each digit is represented by 28 x 28 = 784 values.
>
## STRUCTURE of This Project
> the architecture of autoencdoer is in `pyimagesearch/convautoencoder.py` and for starting the train procedure you can run following command:
```
python train_conv_autoencoder.py
```
furthermore,you can open the `autoencoder-colab.ipynb` in google colab and run it cell by cell,same as below:
> set the matplotlib backend so figures can be saved in the background and import the necessary packages
```
import matplotlib
matplotlib.use("Agg")
from pyimagesearch.convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
```
> initialize the number of epochs to train for and batch size
```
EPOCHS = 25
BS = 32
```
> load the MNIST dataset
```
print("[INFO] loading MNIST dataset...")
((trainX, _), (testX, _)) = mnist.load_data()
```
> add a channel dimension to every image in the dataset, then scale the pixel intensities to the range [0, 1]
```
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
```
> construct our convolutional autoencoder
```
print("[INFO] building autoencoder...")
(encoder, decoder, autoencoder) = ConvAutoencoder.build(28, 28, 1)
opt = Adam(lr=1e-3)
autoencoder.compile(loss="mse", optimizer=opt)
```
> train the convolutional autoencoder
```
H = autoencoder.fit(trainX, trainX,validation_data=(testX, testX),epochs=EPOCHS,batch_size=BS)
```
> construct a plot that plots and saves the training history
```
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
```
> after running this cell, the result of train/validation basis on our dataset will be creating,such as below :
> 
![plot](https://user-images.githubusercontent.com/53394692/111273362-f5878a00-8648-11eb-944a-6e577b0ba1a3.png)
>
> use the convolutional autoencoder to make predictions on the testing images, then initialize our list of output images
```
print("[INFO] making predictions...")
decoded = autoencoder.predict(testX)
outputs = None
```
> loop over our number of output samples
```
for i in range(0,8):
	# grab the original image and reconstructed image
	original = (testX[i] * 255).astype("uint8")
	recon = (decoded[i] * 255).astype("uint8")

	# stack the original and reconstructed image side-by-side
	output = np.hstack([original, recon])

	# if the outputs array is empty, initialize it as the current
	# side-by-side image display
	if outputs is None:
		outputs = output

	# otherwise, vertically stack the outputs
	else:
		outputs = np.vstack([outputs, output])

# save the outputs image to disk
cv2.imwrite("output.png", outputs)
```
> after run this cell you will be seeing,the two columns,left column has different input image,and in right side you see the output image as reconstruction of these images as output of autoencoder,such as below :
> 
![output](https://user-images.githubusercontent.com/53394692/111273922-9ece8000-8649-11eb-9aa2-232f0ad46f13.png)

## License
> [Autoencoders with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/) by Adrian Rosebrock
