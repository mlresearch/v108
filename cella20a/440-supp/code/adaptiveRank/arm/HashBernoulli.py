'''Bernoulli distributed arm.'''

__version__ = "0.1"

from scipy.stats import bernoulli
from numpy import random as rnd

from adaptiveRank.tools.io import c_print
from adaptiveRank.arm.Arm import Arm

class HashBernoulli(Arm):
    """Bernoulli distributed arm"""

    def __init__(self, mu1, mu2, mu0, binary):
        self._means = [mu0, mu1, mu2]
        self._binary = binary

    def __str__(self):
        return "Hash Bernoulli mu1 {}, mu2 {}, mu3 {}".format(self._means[0], self._means[1],self._means[2])

    def draw(self, currentDelay, repIndex):
        expectedReward = self._means[currentDelay]
        return (expectedReward, bernoulli.rvs(expectedReward, random_state = repIndex))

    def computeState(self, currentDelay):
        reward = self._means[currentDelay]
        return reward
