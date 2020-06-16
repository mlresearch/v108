---
title: "“Bring Your Own Greedy”+Max: Near-Optimal 1/2-Approximations for Submodular
  Knapsack"
abstract: The problem of selecting a small-size representative summary of a large
  dataset is a cornerstone of machine learning, optimization and data science. Motivated
  by applications to recommendation systems and other scenarios with query-limited
  access to vast amounts of data, we propose a new rigorous algorithmic framework
  for a standard formulation of this problem as a submodular maximization subject
  to a linear (knapsack) constraint. Our framework is based on augmenting all partial
  Greedy solutions with the best additional item. It can be instantiated with negligible
  overhead in any model of computation, which allows the classic greedy algorithm
  and its variants to be implemented. We give such instantiations in the offline Gready+Max,
  multi-pass streaming Sieve+Max and distributed Distributed Sieve+Max settings. Our
  algorithms give ($1/2-\eps$)-approximation with most other key parameters of interest
  being near-optimal. Our analysis is based on a new set of first-order linear differential
  inequalities and their robust approximate versions. Experiments on typical datasets
  (movie recommendations, influence maximization) confirm scalability and high quality
  of solutions obtained via our framework. Instance-specific approximations are typically
  in the 0.6-0.7 range and frequently beat even the $(1-1/e) \approx 0.63$ worst-case
  barrier for polynomial-time algorithms.
layout: inproceedings
series: Proceedings of Machine Learning Research
id: yaroslavtsev20a
month: 0
tex_title: "“Bring Your Own Greedy”+Max: Near-Optimal 1/2-Approximations for Submodular
  Knapsack"
firstpage: 3263
lastpage: 3274
page: 3263-3274
order: 3263
cycles: false
bibtex_author: Yaroslavtsev, Grigory and Zhou, Samson and Avdiukhin, Dmitrii
author:
- given: Grigory
  family: Yaroslavtsev
- given: Samson
  family: Zhou
- given: Dmitrii
  family: Avdiukhin
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
pdf: http://proceedings.mlr.press/v108/yaroslavtsev20a/yaroslavtsev20a.pdf
extras:
- label: Supplementary PDF
  link: http://proceedings.mlr.press/v108/yaroslavtsev20a/yaroslavtsev20a-supp.pdf
# Format based on citeproc: http://blog.martinfenner.org/2013/07/30/citeproc-yaml-for-bibliographies/
---
