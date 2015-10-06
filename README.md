
The code implements the model (no content) proposed in "Learning Social Network Embeddings for Predicting Information Diffusion - Simon Bourigault, Cedric Lagnier, Sylvain Lamprier, Ludovic Denoyer, Patrick Gallinari, Université Pierre et Marie Curie" published at WSDM 2014. This code is not the one that has been used in the experiments of the article and is provided as if, without using it for large set of experiments...

It is Torch 7 implementation of the model. The model aims at predicting diffusion cascades (typically in social networks)

#Dependencies
* Torch 7
* nn
* nngraph


# Data format

The software needs two input files: a training set of cascades and a testing set. The format of each file is the following:
* one line for each cascade
* each column is [name of the user],[timestamp of the contamination]
* Typically the timestamp of the forst column is one since it corresponds to the source of the cascade

For example, if a twitter message has been posted by John, and the retweeted by Anna 1 hour later, and by Paul 3 hours latter, the cascade will be:
> John,1 Anna,2 Paul,4

Only users that appear at least once in both the train and test files are kept.

Exampels of cascades are given in ''example_train_cascades'' and ''example_test_cascades''

# Command line

The scripts has different arguments:
* training_cascades: the training cascades file
* testing_cascades: the testing cascades file
* outputFile: the final file where embeddings will be stored. Each embedding is a vector, the first column of the file corresponds to the user, the foloowing columns are the embeddings
* learningRate: the learning rate of the SGD algorithm (at the beginning of the process)
* maxEpoch: the number of iterations of the SDG
* evaluationEpoch: evry ''evaluationEpoch' iterations, the software performs a MAP evaluation of the learned embeddings over the training and testing cascades
* uniform: the initialization range for the embeddings
* N: the dimension of the latent space (size of the embeddings)

Contact: ludovic.denoyer@lip6.fr

# TODO

* Version of the model with content
* Baseline methods
* GPU implementation of the distance matrix computation 
