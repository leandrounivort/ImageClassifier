# Image Classifier

This project is an implementation of a deep learning model for classifying images of different dog breeds. It utilizes transfer learning techniques with pre-trained models such as ResNet and EfficientNet.

It uses 20% of the data for validation and a separate test set that does not participate in the training process.  
To prevent overfitting, data augmentation techniques and early stopping are applied, as the model was trained on a relatively small dataset.

A small dataset containing other species, breeds, and elements is used to analyze the behavior of the output neurons in such cases. If none of the neurons exceed the 0.5 threshold, the image is classified as 'Other'.
Despite using a small dataset to train the fully connected (FC) layer, the results show a good differentiation of elements that do not belong to the original breeds.
