''' PIOne policy '''

__version__ = "0.1"

from adaptiveRank.policies.Policy import Policy
from adaptiveRank.tools.io import c_print

class One(Policy):
    def __init__(self):
        self.cIndex = -1
        self.r_star = 0

    def choice(self, arms):
        index = self.cIndex + 1
        if index > self.r_star:
            index = 0

        self.cIndex = index
        return [index]

