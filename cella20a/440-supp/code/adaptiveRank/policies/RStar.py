''' RStar policy '''

__version__ = "0.1"

from adaptiveRank.policies.Policy import Policy
from adaptiveRank.tools.io import c_print

class RStar(Policy):
    def __init__(self, T, MOD):
        self.MOD = MOD
        self.cIndex = -1
        self.T = T

    def initialize(self, r_star):
        self.r_star = r_star
        c_print(4, "RStar, r_star setted to {}".format(r_star))

    def choice(self, arms):
        index = self.cIndex + 1
        if index > self.r_star:
            index = 0

        self.cIndex = index
        return [index]

