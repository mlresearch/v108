''' Ghost policy'''

__version__ = "0.1"

import numpy as np

from adaptiveRank.policies.Policy import Policy
from adaptiveRank.tools.io import c_print

class Ghost(Policy):
    def __init__(self, T, MOD):
        self.cIndex = -1 # last round robin index
        self.r = -1 # rank
        self.MOD =  MOD

    def choice(self, arms):
        # Rank Identification
        if self.r == -1:
            index = arms.argmax()
            # RR termination condition
            if index <= self.cIndex:
                c_print(4, "Ghost rank: {}".format(self.cIndex))
                self.r = self.cIndex
        else: # RR 
            index = self.cIndex + 1
            if index > self.r:
                index = 0

        self.cIndex = index
        return [index]
