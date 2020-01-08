# Neural style transfer with CycleGAN

This repository contains a **Tensorflow** implementation and demonstration of **CycleGAN** technique. _Note that this project is still work in progress and requires some polishing._

The technique performs unpaired image to image translation using conditional GAN's.  It can be trained to translate instances from one domain to another without a one-to-one mapping between the source and target domain.

CycleGAN can be used to solve many interesting problems such as photo-enhancement, image colorization, style transfer, etc. The capabilities are demonstrated in two tasks:
- translation of handwritten digits
- style transfer based on paintings by Leonid Afremov

## How does CycleGAN work
TODO

## Getting Started
TODO

### Prerequisites
- Linux, macOS or Windows
- Python 3 (tested version 3.6)
- CPU or NVIDIA GPU + CUDA CuDNN

### Installation
- Clone this repository
- Setup Python environment (preferably version 3.6) with **pip** command available for this project
- Check _requirements.txt_ with the list of all dependencies. Select appropriate Tensorflow dependency given your configuration.
- Install all dependencies by executing: `pip install -r requirements.txt` into the Python environment.

### Train/test CycleGAN for handwritten digits translation
- Start a **Jupyter server** within your Python environment by executing a command: `jupyter notebook`. Make sure that this command is executed relative to the project directory, so the project's script can be imported with no changes to the code.
- Check the log of a Jupyter server in the console. There should be a server's address. Use it to interact with the server.
- Open '_digit_transformation_' notebook on the server and run all cells. The example should run as it is (it automatically download the data set for you, or cached one will be used)

### Train/test CycleGAN for style transfer
TODO
- The setup is similar to **digits translation example**. However, it requires to prepare a data set. You need to provide two domains 'X' and 'Y'.

## Resources

### References
- https://medium.com/analytics-vidhya/transforming-the-world-into-paintings-with-cyclegan-6748c0b85632
- https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
- https://medium.com/datadriveninvestor/style-transferring-of-image-using-cyclegan-3cc7aff4fe61
- https://www.tensorflow.org/tutorials/generative/cyclegan
- https://arxiv.org/pdf/1703.10593.pdf
- https://towardsdatascience.com/style-transfer-with-gans-on-hd-images-88e8efcf3716
- https://www.tensorflow.org/tutorials/generative/pix2pix
- https://gluon.mxnet.io/chapter14_generative-adversarial-networks/pixel2pixel.html
- https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

### Data
- http://yann.lecun.com/exdb/mnist/
- http://vhosts.eecs.umich.edu/vision/activity-dataset.html
- https://www.afremovpaintings.cz/
- https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
