''' FastPartialOrder plus max-rank policy'''

__version__ = "0.1"

from math import ceil, log, sqrt
from random import choice
from numpy import arange, argpartition, argmax, argsort, array, ones, where, zeros

from adaptiveRank.policies.Policy import Policy
from adaptiveRank.tools.io import c_print

class FPO_UCB(Policy):
    '''FastPartialOrder and MaxRank'''

    def __init__(self, T, tau, delta = 0.1, rounding = 5, MOD = 2, approximate = False, lp = 0, alpha = 1):
        c_print(4, "\nFPO_UCB Init. Tau {}, delta {}".format(tau, delta))
        # Non-stationarity parameters
        self._nArms = 0
        self._tau = tau

        # Running parameters
        self._t = 0 # round iterator
        self._s = 0 # phase iterator
        self._delta = delta
        self._horizon = T
        self._alpha = alpha
        self._rounding = rounding
        self._MOD = MOD
        self._LP = lp # 0 full, 1 arm ordering, 2 rank estimation
        self._APP = approximate

        # Policy "state"
        self._meanArms = []
        self._activeArms = [] # mean ascending sorted arm indexes
        self._ranks = [] # ranks identifiers
        self._nbPullsRanks = []
        self._learnedPO = False

    def setArmMeans(self, means):
        assert self._MOD != 1, "FPO.py() setting Means in a wrong modality"
        self._meanArms = means
        c_print(4, "FPO.py, Setting Arm Means {}".format(self._meanArms))
        return


    def choice(self, arms):
        if self._t == 0: # Data Structures INITIALIZATION
            self._nArms = len(arms)
            self._activeArms = [i for i in range(self._nArms)]
            self._ranks = [i+1 for i in range(self._nArms)]
            self._nbPullsArms = [0] * self._nArms
            self._nbPullsRanks = [0] * self._nArms
            self._cumRwdArms = [0.0] * self._nArms
            self._cumRwdArmDelay = zeros((self._nArms, self._nArms + 1))
            self._nbPullsArmDelay = zeros((self._nArms, self._nArms + 1))

            if self._LP == 2: # Rank Estimation Only
                c_print(4, "FPO.py, JUMPING LEARNING ARM ORDERING: {}".format(self._meanArms))
                self._learnedPO = True
            else:
                self._meanArms = [0.0] * self._nArms
                # Each arm is played once 
                idx = self._bucketing(self._activeArms)
                c_print(4, "FPO.py, choice(): First Pull, round {}, pulling {}".format(self._t, idx))
                return idx

        # Arm Elimination 
        if not self._learnedPO and self._samplingRequired():
            c_print(1, "FPO.py, not ordered DISCARDING on active arms {}".format(self._activeArms))
            self._discarded() # it discards at most a single arm
            idx = self._bucketing(self._activeArms)
            return idx
        else: # Max-Rank
            self._learnedPO = True
            index = list(self._maxrank())
            index.extend(index)
            c_print(1, "FPO.py, choice(): round {} Max_Rank {}\n".format(self._t, index))
            return index


    def _discarded(self):
        for i in range(self._nArms - 1):
            len_activeArms = len(self._activeArms)

            # Due to the bucketing, #activeArms >= tau
            if len_activeArms == self._tau:
                c_print(1, "FPO.py, DISCARDING() TAU not ordered Arms. Active arms {}, Means {}, CB {}".format(self._activeArms, self._meanArms, self._cb()))
                return

            assert self._nArms - self._s == len(self._activeArms), "Inconsistent arm elimination"
            sorted_idx = argsort(self._meanArms)[::-1] #Descending empirical mean order 

            arm_deletion = False # set Canary

            # Gaps computation
            if i != 0:
                gap_l = self._meanArms[sorted_idx[i-1]] - self._meanArms[sorted_idx[i]]
                gap_l = round(gap_l, self._rounding)
            if i != len_activeArms - 1:
                gap_r = self._meanArms[sorted_idx[i]] - self._meanArms[sorted_idx[i+1]]
                gap_r = round(gap_r, self._rounding)

            # Checking Overlapping Conditions
            current_cb = self._cb()
            if i == 0:
                if gap_r > current_cb and sorted_idx[i] in self._activeArms: # Discarding the first index
                    arm_deletion = True
                    c_print(1, "\nFPO.py, DISCARDING(): Gap_right {}, vs CB {}, arm 0 index {}, activeArms {}, sorted_idx {}, means {}".format(gap_r, current_cb, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))
            if i == len_activeArms - 1:
                if gap_l > current_cb and sorted_idx[i] in self._activeArms: # Discarding the last index
                    arm_deletion = True
                    c_print(1, "\nFPO.py, DISCARDING(): Gap_left {}, vs CB {}, last arm index {}, activeArms {}, sorted_idx {}, means {}".format(gap_l, current_cb, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))
            if i != 0 and i != len_activeArms - 1:
                if gap_l > current_cb and gap_r > current_cb and sorted_idx[i] in self._activeArms: # Discarding an index in the middle
                    arm_deletion = True
                    c_print(1, "\nFPO.py, DISCARDING(): Gap_right {}, Gap_left {} vs CB {}, with Index {}, activeArms {}, sorted_idx {}, means {}".format(gap_r, gap_l, current_cb, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))

            if arm_deletion: # Arm deletion based on one of previous cases
                idx = self._activeArms.remove(sorted_idx[i])
                self._s = self._s + 1
                return
        return


    def _samplingRequired(self):
        sorted_idx = argsort(self._meanArms)[::-1]
        len_activeArms = len(self._activeArms)

        for i in range(len_activeArms - 1):
            # Gaps Computations
            if i != 0:
                gap_l =self._meanArms[sorted_idx[i-1]] - self._meanArms[sorted_idx[i]]
                gap_l = round(gap_l, self._rounding)
            if i != len_activeArms - 1:
                gap_r = self._meanArms[sorted_idx[i]] - self._meanArms[sorted_idx[i+1]]
                gap_r = round(gap_r, self._rounding)

            # Checking if more Sampling is required based on Gaps
            current_cb = self._cb()
            if i == 0:
                if gap_r < current_cb:
                    c_print(1, "\nFPO.py, NOT ORDERED(): Gap_right {}, vs CB {}, arm 0 index {}, activeArms {}, sorted_idx {}, means {}".format(gap_r, current_cb, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))
                    return True
            if i == len_activeArms - 1:
                if gap_l < current_cb:
                    c_print(1, "\nFPO.py, NOT ORDERED(): Gap_left {}, vs CB {}, last arm index {}, activeArms {}, sorted_idx {}, means {}".format(gap_l, current_cb, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))
                    return True
            if i != len_activeArms - 1 and i != 0:
                if gap_l < current_cb or gap_r < current_cb:
                    c_print(1, "\nFPO.py, NOT ORDERED(): Gap_right {} Gap_left {}, vs CB {}, arm index {}, activeArms {}, sorted_idx {}, means {}".format(gap_r, gap_l, current_cb, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))
                    return True

        # All arms are Separated
        if not self._learnedPO:
            sorted_idx = argsort(self._meanArms)[::-1]
            c_print(4, "\n===LEARNED ARM ORDERING\nFPO.py, ORDERED(): Learned order relation. Active arms {}, sorted {}, means {}".format(self._activeArms, sorted_idx, self._meanArms))
            self._activeArms = sorted_idx

    def _maxrank(self):
        pulls = array(self._nbPullsRanks)
        zero_idx = where(pulls == 0)[0]
        ucb_values = [0.0] * self._nArms
        if len(zero_idx) > 0:
            index = choice(zero_idx)
        else:
            # For each rank
            for rank in range(self._nArms):
                mean_rank = 0.0
                # For the first rank-arms
                for i in self._activeArms[:(rank+1)]:
                    tmp = self._alpha * self._cumRwdArmDelay[i,rank] / self._nbPullsArmDelay[i,rank]
                    mean_rank = mean_rank + tmp
                ucb_values[rank] = mean_rank /(1+rank) + sqrt((2*log(self._t))/self._nbPullsRanks[rank])
            index = argmax(ucb_values)
        self._nbPullsRanks[index] += 1
        # Additional variables for updating with non-stationarities
        self._pulledRankIndex = index
        self._freezedTime = self._t
        c_print(1, "\nFPO.py, _maxrank(): Pulling rank {} arms {} rank_pulls {} ucb_values {}".format(index+1, self._activeArms[:index+1], self._nbPullsRanks, ucb_values))
        return self._activeArms[:index+1]

    def update(self, arm, rwd, delay):
        c_print(1, "FPO.py, update(): arm {} rwd {} delay {}".format(arm, rwd, delay))
        if not self._learnedPO:
            # Unbiased updates
            if delay == self._tau:
                self._t = self._t + 1
                c_print(4, "FPO.py, update(): unbiased sample")
                self._cumRwdArms[arm] += rwd
                self._nbPullsArms[arm] = self._nbPullsArms[arm] + 1
                self._meanArms[arm] = self._cumRwdArms[arm]/self._nbPullsArms[arm]
        else: # Max Rank stage
            self._t = self._t + 1
            time_gap = self._t - self._freezedTime
            # Windows of acceptance
            if time_gap  > self._pulledRankIndex + 1 and time_gap <= 2 * (self._pulledRankIndex + 1):
                c_print(1, "Storing Arm {} delay {}".format(arm, delay))
                self._cumRwdArmDelay[arm, self._pulledRankIndex] += rwd

                self._nbPullsArmDelay[arm, self._pulledRankIndex] += 1
        return

    def overwriteArmMeans(self, means):
        assert self._LP == 2, "FPO.py, OVERWRITING ARM MEANS IN WRONG MOD"
        self._meanArms = means
        return

    def _bucketing(self, indexes):
        returned_list = []
        len_idx = len(indexes)
        n_full_chunks = int(len_idx/self._tau)
        bucketed = []

        if len(indexes) != self._tau:
            c_print(1, "Number of full chunks {}".format(n_full_chunks))
            for i in range(n_full_chunks):
                c_print(1, "Inserting in full chunks")
                l = min((len_idx, (i+1)*self._tau))
                returned_list += indexes[i*self._tau : l]
                returned_list += indexes[i*self._tau : l]

        # Fill the eventual last partial piece of the list with not pulled active arms
        if ceil(len(indexes)/self._tau) != int(len_idx/self._tau) or self._tau == len(indexes):
            c_print(1, "FPO.py, bucketing(), Completing the last chunk")
            remaining_indexes = indexes[n_full_chunks*self._tau : min(len_idx, (n_full_chunks+1)*self._tau)]
            c_print(1, "FPO.py, bucketing(), Remaining indexes {}".format(remaining_indexes))
            i = 0
            bucketed = remaining_indexes
            while len(bucketed) < self._tau:
                if self._activeArms[i] not in remaining_indexes:
                    bucketed.append(self._activeArms[i])
                    c_print(1, "Index i {}, bucketed {}".format(self._activeArms[i], bucketed))
                i = i + 1
            returned_list += bucketed + bucketed
        c_print(1, "FPO.py, bucketing(), Bucketed List: {}".format(returned_list))

        assert len(returned_list) == 2*self._tau*ceil(len_idx/self._tau), "Inconsistent bucketing"
        return returned_list

    def _cb(self):
        return 2*round(sqrt((log(((len(self._activeArms))*(1+self._t))/(self._delta)))/(1+self._t)), self._rounding)
