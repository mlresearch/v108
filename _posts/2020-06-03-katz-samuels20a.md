---
title: The True Sample Complexity of Identifying Good Arms
abstract: 'We consider two multi-armed bandit problems with $n$ arms: \emph{(i)} given
  an $\epsilon > 0$, identify an arm with mean that is within $\epsilon$ of the largest
  mean and \emph{(ii)} given a threshold $\mu_0$ and integer $k$, identify $k$ arms
  with means larger than $\mu_0$. Existing lower bounds and algorithms for the PAC
  framework suggest that both of these problems require $\Omega(n)$ samples. However,
  we argue that the PAC framework not only conflicts with how these algorithms are
  used in practice, but also that these results disagree with intuition that says
  \emph{(i)} requires only $\Theta(\frac{n}{m})$ samples where $m =  |\{ i : \mu_i
  > \max_{j \in [n]} \mu_j - \epsilon\}|$ and \emph{(ii)} requires $\Theta(\frac{n}{m}k)$
  samples where $m =  |\{ i : \mu_i >  \mu_0 \}|$. We provide definitions that formalize
  these intuitions, obtain lower bounds that match the above sample complexities,
  and develop explicit, practical algorithms that achieve nearly matching upper bounds.'
layout: inproceedings
series: Proceedings of Machine Learning Research
id: katz-samuels20a
month: 0
tex_title: The True Sample Complexity of Identifying Good Arms
firstpage: 1781
lastpage: 1791
page: 1781-1791
order: 1781
cycles: false
bibtex_author: Katz-Samuels, Julian and Jamieson, Kevin
author:
- given: Julian
  family: Katz-Samuels
- given: Kevin
  family: Jamieson
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
pdf: http://proceedings.mlr.press/v108/katz-samuels20a/katz-samuels20a.pdf
extras:
- label: Supplementary PDF
  link: http://proceedings.mlr.press/v108/katz-samuels20a/katz-samuels20a-supp.pdf
# Format based on citeproc: http://blog.martinfenner.org/2013/07/30/citeproc-yaml-for-bibliographies/
---
