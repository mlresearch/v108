''' PIAlt policy '''

__version__ = "0.1"

from adaptiveRank.policies.Policy import Policy
from adaptiveRank.tools.io import c_print

class Alt(Policy):
    def __init__(self):
        self.cIndex = -1
        self.r_star = 1

    def choice(self, arms):
        return [0, 0,  1, 0, 1, 0]
