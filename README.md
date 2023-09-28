# Sparse_VD_keras
Reimplementation of Sparse Variational Dropout in Keras-Core/Keras 3.0, to make it backend-agnostic.
In addition, provides functionality to export/serialize a model which makes use of the custom layers, and implements the ability to use Keras kernel regularizers in the custom layers.
(The above functionality was implemented to allow using orthogonality regularization, to comply with the process described in Improving Word Embedding Using Variational Dropout (FLAIRS 2023) https://journals.flvc.org/FLAIRS/article/view/133326)

Original paper: Variational Dropout Sparsifies Deep Neural Networks (ICML 2017) https://arxiv.org/pdf/1701.05369.pdf
Official repo: https://github.com/bayesgroup/variational-dropout-sparsifies-dnn

Forked from Cerphilly's TensorFlow 2 implementation: https://github.com/Cerphilly/Sparse_VD_tf2

## Requirements

* keras-core >= 0.1.7 (OR keras >= 3.0, once released)