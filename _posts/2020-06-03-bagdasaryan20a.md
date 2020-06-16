---
title: How To Backdoor Federated Learning
abstract: Federated models are created by aggregating model updates submittedby participants.  To
  protect confidentiality of the training data,the aggregator by design has no visibility
  into how these updates aregenerated.  We show that this makes federated learning
  vulnerable to amodel-poisoning attack that is significantly more powerful than poisoningattacks
  that target only the training data.A single or multiple malicious participants can
  use modelreplacement to introduce backdoor functionality into the joint model,e.g.,
  modify an image classifier so that it assigns an attacker-chosenlabel to images
  with certain features, or force a word predictor tocomplete certain sentences with
  an attacker-chosen word.  We evaluatemodel replacement under different assumptions
  for the standardfederated-learning tasks and show that it greatly outperformstraining-data
  poisoning.Federated learning employs secure aggregation to protect confidentialityof
  participants’ local models and thus cannot detect anomalies inparticipants’ contributions
  to the joint model.  To demonstrate thatanomaly detection would not have been effective
  in any case, we alsodevelop and evaluate a generic constrain-and-scale technique
  thatincorporates the evasion of defenses into the attacker’s loss functionduring
  training.
layout: inproceedings
series: Proceedings of Machine Learning Research
id: bagdasaryan20a
month: 0
tex_title: How To Backdoor Federated Learning
firstpage: 2938
lastpage: 2948
page: 2938-2948
order: 2938
cycles: false
bibtex_author: Bagdasaryan, Eugene and Veit, Andreas and Hua, Yiqing and Estrin, Deborah
  and Shmatikov, Vitaly
author:
- given: Eugene
  family: Bagdasaryan
- given: Andreas
  family: Veit
- given: Yiqing
  family: Hua
- given: Deborah
  family: Estrin
- given: Vitaly
  family: Shmatikov
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
pdf: http://proceedings.mlr.press/v108/bagdasaryan20a/bagdasaryan20a.pdf
extras:
- label: Supplementary PDF
  link: http://proceedings.mlr.press/v108/bagdasaryan20a/bagdasaryan20a-supp.pdf
# Format based on citeproc: http://blog.martinfenner.org/2013/07/30/citeproc-yaml-for-bibliographies/
---
