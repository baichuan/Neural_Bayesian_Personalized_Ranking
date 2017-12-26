# Bayesian Personalized Ranking Neural Network
This repository provides a tensorflow implementation of Neural Bayesian Personalized Ranking for top-N item recommendation. 
## Pre-Requisite

* [Python 2.7](https://www.python.org/) 
* [Numpy](http://www.numpy.org/)
* [Tensorflow > 1.3](https://www.tensorflow.org/)

### Basic Usage

#### Example
To run Neural Bayesian Personalized Ranking, execute the following command from the project home directory:<br/>
	``python neural_bpr_v1.py 32 128 0.001 0.01 50``

Current neural structure is input layer -> embedding layer -> one hidden layer with relu activation function -> output layer with BPR loss

To run the Bayesian Personalized Ranking under Matrix Factorization model, execute the following commend from the project home directory: <br/>
	``python bpr_loss_mf.py 32 0.001 0.01 50``  

#### Options
You can check out the hyper-parameter options using:<br/>
	``python neural_bpr_v1.py --help``

### Dataset
Benchmark MovieLens 1M Dataset (https://grouplens.org/datasets/movielens/)

#### Output
The output is pairwise ranking loss, HitRate@10, Normalized Discounted Cumulative Gain@10, Area Under Curve (AUC) in each epoch.
