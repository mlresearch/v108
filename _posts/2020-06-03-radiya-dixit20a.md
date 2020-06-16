---
title: How fine can fine-tuning be?  Learning efficient language models
abstract: 'State-of-the-art performance on language understanding tasks is now achieved
  with increasingly large networks; the current record holder has billions of parameters.  Given
  a language model pre-trained on massive unlabeled text corpora, only very light
  supervised fine-tuning is needed to learn a task: the number of fine-tuning steps
  is typically five orders of magnitude lower than the total parameter count.  Does
  this mean that fine-tuning only introduces \emph{small} differences from the pre-trained
  model in the parameter space?  If so, can one avoid storing and computing an entire
  model for each task?  In this work, we address these questions by using Bidirectional
  Encoder Representations from Transformers (BERT) as an example.  As expected, we
  find that the fine-tuned models are close in parameter space to the pre-trained
  one, with the closeness varying from layer to layer.  We show that it suffices to
  fine-tune only the most critical layers.  Further, we find that there are surprisingly
  many \emph{good} solutions in the set of sparsified versions of the pre-trained
  model.  As a result, fine-tuning of huge language models can be achieved by simply
  setting a certain number of entries in certain layers of the pre-trained parameters
  to zero, saving both task-specific parameter storage and computational cost. '
layout: inproceedings
series: Proceedings of Machine Learning Research
id: radiya-dixit20a
month: 0
tex_title: How fine can fine-tuning be?  Learning efficient language models
firstpage: 2435
lastpage: 2443
page: 2435-2443
order: 2435
cycles: false
bibtex_author: Radiya-Dixit, Evani and Wang, Xin
author:
- given: Evani
  family: Radiya-Dixit
- given: Xin
  family: Wang
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
pdf: http://proceedings.mlr.press/v108/radiya-dixit20a/radiya-dixit20a.pdf
extras:
- label: Supplementary PDF
  link: http://proceedings.mlr.press/v108/radiya-dixit20a/radiya-dixit20a-supp.pdf
# Format based on citeproc: http://blog.martinfenner.org/2013/07/30/citeproc-yaml-for-bibliographies/
---
