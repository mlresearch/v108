---
title: Logistic regression with peer-group effects via inference in higher-order Ising
  models
abstract: Spin glass models, such as the Sherrington-Kirkpatrick, Hopfield and Ising
  models, are all well-studied members of the exponential family of discrete distributions,
  and have been influential in a number of application domains where they are used
  to model correlation phenomena on networks. Conventionally these models have quadratic
  sufficient statistics and consequently capture correlations arising from pairwise
  interactions. In this work we study extensions of these models to models with higher-order
  sufficient statistics, modeling behavior on a social network with peer-group effects.
  In particular, we model binary outcomes on a network as a higher-order spin glass,
  where the behavior of an individual depends on a linear function of their own vector
  of covariates and some polynomial function of the behavior of others, capturing
  peer-group effects. Using a {\em single}, high-dimensional sample from such model
  our goal is to recover the coefficients of the linear function as well as the strength
  of the peer-group effects. The heart of our result is a novel approach for showing
  strong concavity of the log pseudo-likelihood of the model, implying statistical
  error rate of $\sqrt{d/n}$ for the Maximum Pseudo-Likelihood Estimator (MPLE), where
  $d$ is the dimensionality of the covariate vectors and $n$ is the size of the network
  (number of nodes). Our model generalizes vanilla logistic regression  as well as
  the models studied in recent works of  \cite{chatterjee2007estimation,ghosal2018joint,DDP19},
  and our results extend these results to accommodate higher-order interactions.
layout: inproceedings
series: Proceedings of Machine Learning Research
id: daskalakis20a
month: 0
tex_title: Logistic regression with peer-group effects via inference in higher-order
  Ising models
firstpage: 3653
lastpage: 3663
page: 3653-3663
order: 3653
cycles: false
bibtex_author: Daskalakis, Constantinos and Dikkala, Nishanth and Panageas, Ioannis
author:
- given: Constantinos
  family: Daskalakis
- given: Nishanth
  family: Dikkala
- given: Ioannis
  family: Panageas
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
pdf: http://proceedings.mlr.press/v108/daskalakis20a/daskalakis20a.pdf
extras:
- label: Supplementary PDF
  link: http://proceedings.mlr.press/v108/daskalakis20a/daskalakis20a-supp.pdf
# Format based on citeproc: http://blog.martinfenner.org/2013/07/30/citeproc-yaml-for-bibliographies/
---
