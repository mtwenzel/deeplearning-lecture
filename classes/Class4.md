# Class 4: CNNs for Classification

## Optimizers
* Almost overcomplete, available as a paper, as slides, and of course as an online post is [this overview](https://ruder.io/optimizing-gradient-descent/)
* Has many references to original literature and further reading.
## Network components

### Layer Types
- One of the [earlier Deep Learning papers](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) describing typical layers
- [PyTorch Layers](https://pytorch.org/docs/stable/nn.html)
- [Keras Layers](https://keras.io/api/layers/)

### Activation functions, Losses

* [Tutorial at UvA](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html)
* Ramachandran (2017): [Searching for activation functions](https://arxiv.org/abs/1710.05941)
* (Categorial) Cross Entropy
* Dice/Jaccard
* Softmax

### Metrics

* Needed to rate the performance of models in more detail than the loss
* Will be covered within the respective domain topics, e.g. image processing, text processing, ...

## Image analysis in general
- Metrics and how to select them are the topic of [this paper](https://arxiv.org/pdf/2206.01653.pdf).

## CNNs for classification
- [TensorFlow Playground](https://playground.tensorflow.org/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- UvA tutorial on [classification networks](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html)
- Classifiers can't easily signal out-of-distribution examples. [Preview: Energy-based models help.](https://arxiv.org/abs/1912.03263) Cf. Contrastive Divergence!
- Classifier calibration
