---
title: Prophets, Secretaries, and Maximizing the Probability of Choosing the Best
abstract: Suppose a customer is faced with a sequence of fluctuating prices, such
  as for airfare or a product sold by a large online retailer. Given distributional
  information about what price they might face each day, how should they choose when
  to purchase in order to maximize the likelihood of getting the best price in retrospect?
  This is related to the classical secretary problem, but with values drawn from known
  distributions. In their pioneering work, Gilbert and Mosteller [\textit{J. Amer.
  Statist. Assoc. 1966}] showed that when the values are drawn i.i.d., there is a
  thresholding algorithm that selects the best value with probability approximately
  0.58010.5801. However, the more general problem with non-identical distributions
  has remained unsolved.In this paper, we provide an algorithm for the case of non-identical
  distributions that selects the maximum element with probability 1/e1/e, and we show
  that this is tight. We further show that if the observations arrive in a random
  order, this barrier of 1/e1/e can be broken using a static threshold algorithm,
  and we show that our success probability is the best possible for any single-threshold
  algorithm under random observation order. Moreover, we prove that one can achieve
  a strictly better success probability using more general multi-threshold algorithms,
  unlike the non-random-order case. Along the way, we show that the best achievable
  success probability for the random-order case matches that of the i.i.d. case, which
  is approximately 0.58010.5801, under a “no-superstars” condition that no single
  distribution is very likely ex ante to generate the maximum value. We also extend
  our results to the problem of selecting one of the kk best values.One of the main
  tools in our analysis is a suitable “Poissonization” of random order distributions,
  which uses Le Cam’s theorem to connect the Poisson binomial distribution with the
  discrete Poisson distribution. This approach may be of independent interest.
layout: inproceedings
series: Proceedings of Machine Learning Research
id: esfandiari20a
month: 0
tex_title: Prophets, Secretaries, and Maximizing the Probability of Choosing the Best
firstpage: 3717
lastpage: 3727
page: 3717-3727
order: 3717
cycles: false
bibtex_author: Esfandiari, Hossein and Hajiaghayi, MohammadTaghi and Lucier, Brendan
  and Mitzenmacher, Michael
author:
- given: Hossein
  family: Esfandiari
- given: MohammadTaghi
  family: Hajiaghayi
- given: Brendan
  family: Lucier
- given: Michael
  family: Mitzenmacher
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
pdf: http://proceedings.mlr.press/v108/esfandiari20a/esfandiari20a.pdf
extras:
- label: Supplementary PDF
  link: http://proceedings.mlr.press/v108/esfandiari20a/esfandiari20a-supp.pdf
# Format based on citeproc: http://blog.martinfenner.org/2013/07/30/citeproc-yaml-for-bibliographies/
---
