# Class 3: Back Propagation, SGD, Losses

## Contrastive Divergence
- Used in RBMs, cf. Section "Learning Weights" in the [RBM Tutorial](http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/)
- Also a kind of unsupervised training paradigm (see [Class 8](Class8.md))
## Backpropagation

* Original paper (Hinton et al, 1986): [Learning Internal Representations by Error Propagation](https://apps.dtic.mil/sti/citations/ADA164453)

## SGD
* Can wrap any algorithm like BP or CD.

## Activation functions, Losses

* [Tutorial at UvA](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html)
* Ramachandran (2017): [Searching for activation functions](https://arxiv.org/abs/1710.05941)
* (Categorial) Cross Entropy
* Dice/Jaccard
* Softmax

## Metrics

* Needed to rate the performance of models in more detail than the loss
* Will be covered within the respective domain topics, e.g. image processing, text processing, ...