# Class 3: Back Propagation, SGD, Losses

## Neural Networks Intro

### Neural Networks: Architectures and Bias

- Paper on [Inductive Biases for Deep Learning of Higher-Level Cognition](https://arxiv.org/abs/2011.15091)
- Blog post on [inductive bias in common network types](https://towardsdatascience.com/the-inductive-bias-of-ml-models-and-why-you-should-care-about-it-979fe02a1a56)
- Battaglia paper on [inductive bias in graph networks](https://arxiv.org/abs/1806.01261)

## Training Algorithms
### Contrastive Divergence
- Used in RBMs, cf. Section "Learning Weights" in the [RBM Tutorial](http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/) or [this YouTube video series](https://youtu.be/p4Vh_zMw-HQ). Compact PyTorch code for a RMB can be found [here](https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/)
- Also a kind of unsupervised training paradigm (see [Class 8](Class8.md))

### Backpropagation

* Original paper (Hinton et al, 1986): [Learning Internal Representations by Error Propagation](https://apps.dtic.mil/sti/citations/ADA164453)
* [Online tutorial](http://neuralnetworksanddeeplearning.com/chap2.html)
* [BP example step by step](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)

## Optimizers
* SGD - Can wrap any algorithm like BP or CD.
* ADAM
* ADAGrad

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