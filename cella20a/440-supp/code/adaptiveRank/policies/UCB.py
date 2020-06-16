''' UCB policy'''

__version__ = "0.1"

from numpy import argmax, where, zeros
from random import choice
from math import log, sqrt

from adaptiveRank.policies.Policy import Policy
from adaptiveRank.tools.io import c_print

class UCB(Policy):
    '''UCB1 Policy'''

    def __init__(self, T, MOD):
        self.t = 0
        self.MOD = MOD
        self.T = T

    def choice(self, arms):
        # New round
        self.t =  self.t + 1

        if self.t > 1:
            not_pulled_idx = where(self._nbPulls==0)
            c_print(self.MOD, "Round {} Not pulled Indexes: {}".format(self.t, not_pulled_idx[0]))
            if len(not_pulled_idx[0]) > 0:
                index = choice(not_pulled_idx[0])
            else:
                ucb_values = zeros(len(arms))
                for i in range(len(arms)):
                    ucb_values[i] = (self._cumRwds[i]/self._nbPulls[i]) + sqrt((2*log(self.t))/self._nbPulls[i])
                index = argmax(ucb_values)
                c_print(self.MOD, "Round {} UCB {}".format(self.t, ucb_values))
        else: #First run only
            self._cumRwds = zeros(len(arms))
            self._nbPulls = zeros(len(arms))
            index = choice(range(len(arms)))

        c_print(self.MOD, "Round {} CumRwds {} NbPulls {}".format(self.t, self._cumRwds, self._nbPulls))
        c_print(self.MOD, "Index: {}\n".format(index))

        self._nbPulls[index] = self._nbPulls[index] + 1
        return [index]

    def update(self, arm, rwd, delay):
        self._cumRwds[arm] = self._cumRwds[arm] + rwd
