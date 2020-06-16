---
title: Adaptive, Distribution-Free Prediction Intervals for Deep Networks
abstract: The machine learning literature contains several constructions for prediction
  intervals that are intuitively reasonable but ultimately ad-hoc in that they do
  not come with provable performance guarantees. We present methods from the statistics
  literature that can be used efficiently with neural networks under minimal assumptions
  with guaranteed performance. We propose a neural network that outputs three values
  instead of a single point estimate and optimizes a loss function motivated by the
  standard quantile regression loss. We provide two prediction interval methods with
  finite sample coverage guarantees solely under the assumption that the observations
  are independent and identically distributed. The first method leverages the conformal
  inference framework and provides average coverage. The second method provides a
  new, stronger guarantee by conditioning on the observed data. Lastly, our loss function
  does not compromise the predictive accuracy of the network like other prediction
  interval methods. We demonstrate the ease of use of our procedures as well as its
  improvements over other methods on both simulated and real data. As most deep networks
  can easily be modified by our method to output predictions with valid prediction
  intervals, its use should become standard practice, much like reporting standard
  errors along with mean estimates.
layout: inproceedings
series: Proceedings of Machine Learning Research
id: kivaranovic20a
month: 0
tex_title: Adaptive, Distribution-Free Prediction Intervals for Deep Networks
firstpage: 4346
lastpage: 4356
page: 4346-4356
order: 4346
cycles: false
bibtex_author: Kivaranovic, Danijel and Johnson, Kory D. and Leeb, Hannes
author:
- given: Danijel
  family: Kivaranovic
- given: Kory D.
  family: Johnson
- given: Hannes
  family: Leeb
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
pdf: http://proceedings.mlr.press/v108/kivaranovic20a/kivaranovic20a.pdf
extras:
- label: Supplementary PDF
  link: http://proceedings.mlr.press/v108/kivaranovic20a/kivaranovic20a-supp.pdf
# Format based on citeproc: http://blog.martinfenner.org/2013/07/30/citeproc-yaml-for-bibliographies/
---
