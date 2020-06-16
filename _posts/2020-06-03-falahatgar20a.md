---
title: Towards Competitive N-gram Smoothing
abstract: 'N-gram models remain a fundamental component of language modeling. In data-scarce
  regimes, they are a strong alternative to neural models. Even when not used as-is,
  recent work shows they can regularize neural models. Despite this success, the effectiveness
  of one of the best N-gram smoothing methods, the one suggested by Kneser and Ney
  (1995), is not fully understood. In the hopes of explaining this performance, we
  study it through the lens of competitive distribution estimation: the ability to
  perform as well as an oracle aware of further structure in the data. We first establish
  basic competitive properties of Kneser-Ney smoothing. We then investigate the nature
  of its backoff mechanism and show that it emerges from first principles, rather
  than being an assumption of the model. We do this by generalizing the Good-Turing
  estimator to the contextual setting. This exploration leads us to a powerful generalization
  of Kneser-Ney, which we conjecture to have even stronger competitive properties.
  Empirically, it significantly improves performance on language modeling, even matching
  feed-forward neural models. To show that the mechanisms at play are not restricted
  to language modeling, we demonstrate similar gains on the task of predicting attack
  types in the Global Terrorism Database.'
layout: inproceedings
series: Proceedings of Machine Learning Research
id: falahatgar20a
month: 0
tex_title: Towards Competitive N-gram Smoothing
firstpage: 4206
lastpage: 4215
page: 4206-4215
order: 4206
cycles: false
bibtex_author: Falahatgar, Moein and Ohannessian, Mesrob and Orlitsky, Alon and Pichapati,
  Venkatadheeraj
author:
- given: Moein
  family: Falahatgar
- given: Mesrob
  family: Ohannessian
- given: Alon
  family: Orlitsky
- given: Venkatadheeraj
  family: Pichapati
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
pdf: http://proceedings.mlr.press/v108/falahatgar20a/falahatgar20a.pdf
extras:
- label: Supplementary PDF
  link: http://proceedings.mlr.press/v108/falahatgar20a/falahatgar20a-supp.pdf
# Format based on citeproc: http://blog.martinfenner.org/2013/07/30/citeproc-yaml-for-bibliographies/
---
