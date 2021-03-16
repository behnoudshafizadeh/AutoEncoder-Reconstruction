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




























## License
> [Autoencoders with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/) by Adrian Rosebrock
