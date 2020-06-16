'''Bernoulli distributed arm.'''

__version__ = "0.1"

from scipy.stats import bernoulli
from numpy import random as rnd

from adaptiveRank.tools.io import c_print
from adaptiveRank.arm.Arm import Arm

TEST = True

class Bernoulli(Arm):
    """Bernoulli distributed arm"""

    def __init__(self, mean, gamma, minDelay, maxDelay, approximate):
        self._mean = mean
        self._gamma = gamma
        self._minDelay = minDelay
        self._maxDelay = maxDelay
        self._approximate = approximate


    def __str__(self):
        return "Bernoulli arm. mu: {} gamma: {} min_delay {}: max_delay: {}".format(self._mean, self._gamma, self._minDelay, self._maxDelay)

    def draw(self, currentDelay, rep_index):
        expectedReward = self._mean
        if currentDelay != 0 and currentDelay <= self._maxDelay and currentDelay >= self._minDelay:
            c_print(1, "Discounting")
            expectedReward *= (1 - self._gamma**currentDelay)
        return (expectedReward, bernoulli.rvs(expectedReward, random_state = rep_index))

    def computeState(self, currentDelay):
        expectedReward = self._mean
        if currentDelay <= self._maxDelay and currentDelay >= self._minDelay:
            expectedReward = self._mean * (1.0 - (self._gamma**currentDelay))
            c_print(1, "Arm.py, computeState(), disc mean {}".format(expectedReward))
        else:
            c_print(1, "Arm.py, computeState(), exp mean {}".format(expectedReward))
        return expectedReward
