# End-to-end convolutional selective autoencoder for Soybean Cyst Nematode Eggs detection and counting.

This code runs trains and tests convolutional selective autoencoder models. The architecture is successfully applied on the problem of detecting soybean cysts nematode eggs on microscopic image plates whose preliminary work is described in [An end-to-end convolutional selective autoencoder
approach to Soybean Cyst Nematode eggs detection](https://arxiv.org/pdf/1603.07834.pdf). The architecture is designed for problems with rare objects of interest, and to discriminate highly similar objects  The inference part also counts the number of eggs using a connected components principle.

## Getting Started

These instructions will get you a copy of the project and ensure it runs properly on your local machine for development and testing purposes.

## Prerequisites and Installations

The code has been tested in the following versions of library
* https://raw.githubusercontent.com/dnouri/nolearn/0.6.0/requirements.txt
* [anaconda-Python 2.7](https://conda.io/docs/user-guide/install/index.html)
* Theano version 0.7

## SCN detection samples
In the microscopic image samples below, Soybean Cyst Nematode (SCN) eggs are marked by pathologists


## Preliminaries

* gpu0 - command for specifying the gpu to use. 
* train_number is a unique identifier for saving a new training model e.g. 1, 2, 3,...etc. 
* threshold_value is a value between 0 and 255 that helps in the postprocessing for objects and non-object level.
* number_of_epochs is the variable number of runs of the training required e.g. 10, 50, 100, etc.

## Training a model
* Function lets you train a new model
```
gpu0 python CSAE.py Train_model train_number threshold_val number_of_epochs
```

## Validate trained model
* Function lets you validate an existing model, visulalizes the weights and biases and the error plots 
```
gpu0 python CSAE.py Validate_model train_number threshold_val number_of_epochs
```

## Model inference
* Function lets you Test on unseen samples related to model training.
```
gpu0 python CSAE.py Test_model train_number threshold_val number_of_epochs
```

## Trained model

* A trained model to test the algorithm can be downloaded from [Trained model](https://iastate.box.com/s/u1ddojb3ty9z4fiayeduuwbywuszhdi7).
* The raw training image will be released soon after permissions are granted.
* Example Test images and results with different thresholds can be found in SCNDatasets/
## Deployment to New dataset

* Interact with the settings for users from line 69 to line 95
* Repeat the Model training, validation and inference above

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Daniel Nouri. 2014. nolearn: scikit-learn compatible neural network library https://github.com/dnouri/nolearn


