# LBCM
Code for running experiments using the LBCM 

## Environment Set Up
This code was made to run on `Python 3.7+` and I recommend using `conda` to create an environment. 

You will need the following packages (which I recommend acquiring using `conda-forge`)
```
- numpy
- scipy
- pot (Python OT)
- matplotlib
- cvxopt
```
In addition you will need access to the MNIST dataset which is publicly available. This can be done using either the `mnist` package which is available [here](https://pypi.org/project/mnist/) or by using `tensorflow`. The default location that it is expected to be is inside a folder named `mnist` in the same folder as the code.

## Demos

To run the demos, activate the environment with the dependencies installed and run either `python mnist_demo.py` or `python gauss_demo.py` or `python mnist_diff_base.py`

### The MNIST Demo

This compares the recovery of a corrupted MNIST digit using either the BCM or the LBCM with two different base measures. It is set to use 20 references on the digit '4'. The two base measures considered are 1. The uniform image over the 28x28 grid or 2. The (Convolutional) Barycenter of the reference images.

### The Different Base Demo

This runs a further version of the MNIST Demo using different and more exotic choices of the base measure.

### The Gaussian Demo

This demo compares the recovery of a Gaussian covariance matrix from samples. There are four recovery methods considered: 1. The empirical covariance, 2. The method considered in [The BCM paper](https://proceedings.mlr.press/v162/werenski22a/werenski22a.pdf) 3. The LBCM with identity reference measure 4. The LBCM with the barycenter of the references. The default settings are 10 refefences and 10 dimensions.
