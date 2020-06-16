---
title: Approximate Cross-Validation in High Dimensions with Guarantees
abstract: Leave-one-out cross-validation (LOOCV) can be particularly accurate among
  cross-validation (CV) variants for machine learning assessment tasks – e.g., assessing
  methods’ error or variability. But it is expensive to re-fit a model $N$ times for
  a dataset of size $N$. Previous work has shown that approximations to LOOCV can
  be both fast and accurate – when the unknown parameter is of small, fixed dimension.
  But these approximations incur a running time roughly cubic in dimension – and we
  show that, besides computational issues, their accuracy dramatically deteriorates
  in high dimensions. Authors have suggested many potential and seemingly intuitive
  solutions, but these methods have not yet been systematically evaluated or compared.
  We find that all but one perform so poorly as to be unusable for approximating LOOCV.
  Crucially, though, we are able to show, both empirically and theoretically, that
  one approximation can perform well in high dimensions – in cases where the high-dimensional
  parameter exhibits sparsity. Under interpretable assumptions, our theory demonstrates
  that the problem can be reduced to working within an empirically recovered (small)
  support. This procedure is straightforward to implement, and we prove that its running
  time and error depend on the (small) support size even when the full parameter dimension
  is large.
layout: inproceedings
series: Proceedings of Machine Learning Research
id: stephenson20a
month: 0
tex_title: Approximate Cross-Validation in High Dimensions with Guarantees
firstpage: 2424
lastpage: 2434
page: 2424-2434
order: 2424
cycles: false
bibtex_author: Stephenson, William and Broderick, Tamara
author:
- given: William
  family: Stephenson
- given: Tamara
  family: Broderick
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
pdf: http://proceedings.mlr.press/v108/stephenson20a/stephenson20a.pdf
extras:
- label: Supplementary PDF
  link: http://proceedings.mlr.press/v108/stephenson20a/stephenson20a-supp.pdf
# Format based on citeproc: http://blog.martinfenner.org/2013/07/30/citeproc-yaml-for-bibliographies/
---
