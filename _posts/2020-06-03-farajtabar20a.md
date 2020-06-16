---
title: Orthogonal Gradient Descent for Continual Learning
abstract: Neural networks are achieving state of the art and sometimes super-human
  performance on learning tasks across a variety of domains. Whenever these problems
  require learning in a continual or sequential manner, however, neural networks suffer
  from the problem of catastrophic forgetting; they forget how to solve previous tasks
  after being trained on a new task, despite having the essential capacity to solve
  both tasks if they were trained on both simultaneously. In this paper, we propose
  to address this issue from a parameter space perspective and study an approach to
  restrict the direction of the gradient updates to avoid forgetting previously-learned
  data. We present the Orthogonal Gradient Descent (OGD) method, which accomplishes
  this goal by projecting the gradients from new tasks onto a subspace in which the
  neural network output on previous task does not change and the projected gradient
  is still in a useful direction for learning the new task. Our approach utilizes
  the high capacity of a neural network more efficiently and does not require storing
  the previously learned data that might raise privacy concerns. Experiments on common
  benchmarks reveal the effectiveness of the proposed OGD method.
layout: inproceedings
series: Proceedings of Machine Learning Research
id: farajtabar20a
month: 0
tex_title: Orthogonal Gradient Descent for Continual Learning
firstpage: 3762
lastpage: 3773
page: 3762-3773
order: 3762
cycles: false
bibtex_author: Farajtabar, Mehrdad and Azizan, Navid and Mott, Alex and Li, Ang
author:
- given: Mehrdad
  family: Farajtabar
- given: Navid
  family: Azizan
- given: Alex
  family: Mott
- given: Ang
  family: Li
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
pdf: http://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf
extras:
- label: Supplementary PDF
  link: http://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a-supp.pdf
# Format based on citeproc: http://blog.martinfenner.org/2013/07/30/citeproc-yaml-for-bibliographies/
---
