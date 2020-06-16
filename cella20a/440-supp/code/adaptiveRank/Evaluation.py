'''Utility class for the performance evaluation'''

__version__ = "0.1"

import numpy as np
from adaptiveRank.tools.io import c_print
from joblib import Parallel, delayed

import sys
sys.maxsize = 1000000 # Avoid truncations in print

def parallel_repetitions(evaluation, policy, horizon, i):
    c_print(2, "EVALUATION: parallel_repetition index {}, T: {}".format(i+1, horizon))
    result = evaluation.environment.play(policy, horizon, i)
    return (i,result)

class Evaluation:
    def __init__(self, env, pol, horizon, policyName, nbRepetitions ):
        ''' Initialized in the run.py file.'''

        # Associated learning problem: policy, environment
        self.environment = env
        self.policy = pol

        # Learning problem parameters: horizon, policy name, nbRepetitions 
        self.horizon = horizon
        self.polName = policyName
        self.nbRepetitions = nbRepetitions

        # Data Structurs to store the results of different reward samples
        self.rewards = np.zeros(self.nbRepetitions)
        self.cumSumRwd = np.zeros((self.nbRepetitions, self.horizon))

        c_print(4,"===Evaluation.py, INIT: {} over {} rounds for {} nbRepetitions".format(self.polName, self.horizon, self.nbRepetitions))

        # Parallel call to the policy run over the number of repetitions
        with Parallel(n_jobs = self.nbRepetitions) as parallel:
            repetitionIndex_results = parallel(delayed(parallel_repetitions)(self, self.policy, self.horizon, i) for i in range(nbRepetitions))

        # Results extrapolation
        for i, result in repetitionIndex_results:
            self.rewards[i] = result.getReward() # Over the flattened array
            self.cumSumRwd[i] = result.getCumSumRwd()
            self.nbArms = result.getNbArms()

        # Additional Result Visualization 
        if len(repetitionIndex_results) == 1:
            c_print(1,"Evaluation.py\n{}".format(repr(repetitionIndex_results[0][1])))

        c_print(2, "EVALUATION: End iteration over {} repetitions for {}".format(nbRepetitions, policyName))

        # Averaged best Expectation
        self.meanReward = np.mean(self.rewards)
        self.meanCumSumRwd = np.mean(self.cumSumRwd, axis = 0)
        self.stdCumSumRwd = np.std(self.cumSumRwd, axis = 0)

        # Results visualization
        c_print(4, "Evaluation.py, Pol: {}, Rewards: {} Average: {}".format(policyName, self.rewards, self.meanReward))
        c_print(2, "Evaluation.py, Pol: {}, Cumulative Rewards:\n{}".format(policyName, self.cumSumRwd))
        c_print(2, "Evaluation.py, Pol: {}, Mean CumulativeReward:\n{}".format(policyName, self.meanCumSumRwd))
        c_print(2, "Evaluation.py, Pol: {}, Std CumulativeReward:\n{}".format(policyName, self.stdCumSumRwd))

        self.result = (policyName, self.meanCumSumRwd, self.stdCumSumRwd)

    def getResults(self):
        return self.result 

    def getNbArms(self):
        return self.nbArms
