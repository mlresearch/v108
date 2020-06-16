---
title: Deterministic Decoding for Discrete Data in Variational Autoencoders
abstract: Variational autoencoders are prominent generative models for modeling discrete
  data. However, with flexible decoders, they tend to ignore the latent codes.  In
  this paper, we study a VAE model with a deterministic decoder (DD-VAE) for sequential
  data that selects the highest-scoring tokens instead of sampling. Deterministic
  decoding solely relies on latent codes as the only way to produce diverse objects,
  which improves the structure of the learned manifold. To implement DD-VAE, we propose
  a new class of bounded support proposal distributions and derive Kullback-Leibler
  divergence for Gaussian and uniform priors. We also study a continuous relaxation
  of deterministic decoding objective function and analyze the relation of reconstruction
  accuracy and relaxation parameters. We demonstrate the performance of DD-VAE on
  multiple datasets, including molecular generation and optimization problems.
layout: inproceedings
series: Proceedings of Machine Learning Research
id: polykovskiy20a
month: 0
tex_title: Deterministic Decoding for Discrete Data in Variational Autoencoders
firstpage: 3046
lastpage: 3056
page: 3046-3056
order: 3046
cycles: false
bibtex_author: Polykovskiy, Daniil and Vetrov, Dmitry
author:
- given: Daniil
  family: Polykovskiy
- given: Dmitry
  family: Vetrov
date: 2020-06-03
address: 
publisher: PMLR
container-title: Proceedings of the Twenty Third International Conference on Artificial
  Intelligence and Statistics
volume: '108'
genre: inproceedings
issued:
  date-parts:
  - 2020
  - 6
  - 3
pdf: http://proceedings.mlr.press/v108/polykovskiy20a/polykovskiy20a.pdf
extras:
- label: Supplementary PDF
  link: http://proceedings.mlr.press/v108/polykovskiy20a/polykovskiy20a-supp.pdf
# Format based on citeproc: http://blog.martinfenner.org/2013/07/30/citeproc-yaml-for-bibliographies/
---
