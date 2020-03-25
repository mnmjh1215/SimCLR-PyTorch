# SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

Unofficial PyTorch implementation of [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) for CIFAR-10 dataset.

This implementation is only meant to showw that SimCLR framework, and Contrastive Learning in general, works, so it is not fully optimized.

## Result

I evaluated SimCLR on CIFAR-10 dataset using linear evaluation method.

I trained SimCLR on CIFAR-10 train dataset for 100 epochs, using 10% of train dataset as validation dataset. (I did not conduct any hyperparameter search)

Below is the result of training linear classifier on training set, evaluated on test set.

| Feature Extractor       | number of features | test accuracy |
|-------------------------|--------------------|---------------|
| PCA                     | 512                | 40.7%         |
| ResNet-18 (with SimCLR) | 512                | 83.63%        |


