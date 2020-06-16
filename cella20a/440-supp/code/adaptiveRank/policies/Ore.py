''' FastPart:ialOrder plus max-rank policy'''

__version__ = "0.1"

from math import ceil, log, sqrt
from random import choice
from numpy import arange, argpartition, argmax, argsort, array, ones, where, zeros

from adaptiveRank.policies.Policy import Policy
from adaptiveRank.tools.io import c_print

class ORE2(Policy):
    '''Ordering and Rank Estimation via Elimination'''

    def __init__(self, T, tau, delta = 0.1, shrink = 1, rounding = 5, MOD = 2, approximate = False, lp = 0):
        c_print(4, "\nORE2 Init. Tau {}, delta {}, LP {}".format(tau, delta, lp))
        # Non-stationarity parameters
        self._tau = tau

        # Running parameters
        self.horizon = T # horizon
        self.delta = delta # confidence
        self._shrink = shrink
        self._rounding = rounding # rounding approximation
        self.MOD = MOD
        self.APP = approximate
        self.LP = lp

        # Policy "state"
        self._t = 0 # round iterator, necessary for filtering biased rewards
        self._r = 0 # stage iterator in terms of rounds
        self._s = 0 # stage iterator in terms of elimination

        # Policy "state": RANKS Data Structures
        self._activeArms = [] # mean ascending sorted arm indexes
        self._activeRanks = [] # rank indexes
        self._learnedPO = False # canary stating if the arm ordering was learnt
        self._learnedRank = False # canary stating if the rank was learnt
        self._jump_list = [] # binary list denoting skipping rounds due to calibration
        self._jump_rank = [] # rank list denoting current rank to be updated after calibration


    def choice(self, arms):
        # First Round
        if self._t == 0: # Policy "state" Data Structures Construction
            self._nArms = len(arms)
            self._activeArms = [i for i in range(self._nArms)]
            self._activeRanks = [i for i in range(self._nArms)]
            self._nbPullsArms = [0] * self._nArms
            self._nbPullsRanks = [0] * self._nArms
            self._cumRwdArms = [0.0] * self._nArms
            self._cumRwdArmDelay = zeros((self._nArms, self._nArms))
            self._nbPullsArmDelay = zeros((self._nArms, self._nArms))
            self._cumRwdRanks = [0.0] * self._nArms
            self._meanRanks = [0.0] * self._nArms

            # Rank Estimation Problem
            if self.LP == 2:
                self._learnedPO = True
                sorted_idx = argsort(self._meanArms)[::-1]
                c_print(4, "ORE.py, JUMPING ARM ORDERING ESTIMATION, given means {} sorted indexes {}".format(self._meanArms, sorted_idx))
                # Variable setting for the next phase
                self._r = 1
                self._s = 0
                self._activeArms = sorted_idx
            else:# Arm Ordering initialization
                self._meanArms = [0.0] * self._nArms
                # Each arm is played once 
                idx = self._bucketing(self._activeArms)
                c_print(self.MOD, "ORE.py, CHOICE(): First Pull, round {}, pulling {}".format(self._t, idx))
                self._r = self._r + 1
                assert self._r == 1, "Wrong sampling counter definition"
                return idx

        # Arm Elimination
        if not self._learnedPO and self._samplingRequired():
            c_print(1, "PO.py, not ordered DISCARDING on active arms {}".format(self._activeArms))
            self._discarded() # it discards at most a single arm
            idx = self._bucketing(self._activeArms)
            self._r = self._r + 1
            return idx
        else: # Rank Elimination
            self._learnedPO = True
            self._roundLearnedPO = self._t
            index = list()
            jump_list = list()
            jump_rank = list()
            # Stage 1: Sampling all active arms
            self._freezedTime = self._t
            nbAppends = max(int(self._Ts() / (self._r * len(self._activeRanks) * self._shrink)), 2)

            # Output
            if len(self._activeRanks) > 1:
                c_print(1, "ORE.py, CHOICE INIT RANK ELIMINATION {}-ROUND with {} appends per rank".format(self._r, nbAppends))
                c_print(1, "ORE.py, RANKS MEANS {}, Nb Pulls: {}".format(self._meanRanks, self._nbPullsRanks))
                c_print(1, "ORE.py, choice: round {}, Active Ranks {}\n".format(self._t, self._activeRanks))

            # Playing the less played rank among the active ones
            active_ranks_pulls = [self._nbPullsRanks[i] for i in self._activeRanks]
            rank_id = self._activeRanks[active_ranks_pulls.index(min(active_ranks_pulls))]
            # Additional variables for updating with non-stationarities
            self._pulledRankIndex = rank_id
            self._nbPullsRanks[rank_id] += 1
            # List extension: calibration + Ts
            c_print(1, "\nORE.py, CHOICE pulled rank {}".format(rank_id))
            tmp_index = list(self._activeArms[:rank_id+1])
            jump_list += [0] * (rank_id+1)
            jump_rank += [rank_id] * (rank_id+1)
            index += tmp_index # Rank Calibration
            for _ in range(nbAppends): # Effective pulls
                index += tmp_index
                jump_rank += [rank_id] * (rank_id+1)
                jump_list += [1] * (rank_id+1)

            c_print(1, "ORE.py, CHOICE rank_id: {} with {} appends, active ranks pulls {}".format(rank_id, nbAppends, active_ranks_pulls))

            # NO RANK ELIMINATION at first round or within a window of active ranks pulls
            if self._t == 0 or min(active_ranks_pulls)!= max(active_ranks_pulls):
                self._jump_list = jump_list
                self._jump_rank = jump_rank
                return index

            # Stage 2: Rank Elimination
            c_print(1, "ORE.py Start Rank Elimination with {} active ranks".format(len(self._activeRanks)))
            self._r = self._r + 1
            if len(self._activeRanks) > 1:
                self._rankElimination()
            self._jump_list = jump_list
            self._jump_rank = jump_rank
            return index


    def _rankElimination(self):
        assert len(self._activeRanks) == self._nArms - self._s, "Incoherent Rank Elimination"
        # Update Ranks Statistics
        for rank in self._activeRanks:
            mean_rank = 0.0
            for i in self._activeArms[:(rank+1)]:
                assert self._nbPullsArmDelay[i, rank] != 0, "Wrong variable update nbPullsArmDelay"
                tmp = self._cumRwdArmDelay[i,rank] / self._nbPullsArmDelay[i,rank]
                mean_rank = mean_rank + tmp
            self._meanRanks[rank] = mean_rank / (rank + 1)

        # Update the set of Active Ranks 
        max_rank_id = argmax(self._meanRanks)
        for rank_id in self._activeRanks:
            ranks_gap = self._meanRanks[max_rank_id] - self._meanRanks[rank_id]
            # Rank Elimination: eliminating up to 1 rank per round
            if ranks_gap > self._cb():
                c_print(1,"\nRANK ELIMINATION(), Eliminating Rank {}, vs {}, cb {}, gap {}, means {}".format(rank_id, max_rank_id, self._cb(), ranks_gap, self._meanRanks))
                self._s = self._s + 1
                self._activeRanks.remove(rank_id)
                if len(self._activeRanks) == 1:
                    c_print(4, "RANK Elimination(), LEARNED RANK {}".format(self._activeRanks[0]))
        return 


    def update(self, arm, rwd, delay):
        c_print(1, "ORE.py, update(): arm {} rwd {} delay {}".format(arm, rwd, delay))
        if not self._learnedPO:
            # Unbiased updates
            if delay == self._tau:
                self._t = self._t + 1
                c_print(1, "ORE.py, update(): unbiased sample")
                self._cumRwdArms[arm] += rwd
                self._nbPullsArms[arm] = self._nbPullsArms[arm] + 1
                self._meanArms[arm] = self._cumRwdArms[arm]/self._nbPullsArms[arm]
        else: # Max Rank stage
            time_gap = self._t - self._freezedTime
            self._t = self._t + 1
            pulledRankIndex = self._jump_rank[time_gap]
            # Windows of acceptance
            if self._jump_list[time_gap]:
                c_print(1, "Storing rwd {} for Arm {} delay {}".format(rwd, arm, pulledRankIndex))
                self._cumRwdArmDelay[arm, pulledRankIndex] += rwd
                self._nbPullsArmDelay[arm, pulledRankIndex] += 1
            else:
                c_print(1, "Discarding rwd {} for Arm {} delay {}".format(rwd, arm, pulledRankIndex))
        return


    def _cb(self):
        # Confidence bounds definition depending on the stage
        if self._learnedPO: # CB for Rank Elimination
            return 0.001*round(sqrt( log(((self._Ts() + self._nArms) * self._nArms)/self.delta) * (self._nArms / (2 * self._Ts()))), self._rounding)
        else: # CB for Arm Ordering Estimation
            return round(sqrt(log((2 * self._nArms * self._r * (self._r+1))/(self.delta)) * (1/(10*self._r))), self._rounding)


    def _Ts(self):
        return int((self.horizon) ** (1 - 2**(-self._r)))


    def _times(self):
        return max(int(self._Ts() / (self._r * len(self._activeRanks))), 1)


    def _samplingRequired(self):
        sorted_idx = argsort(self._meanArms)[::-1] # arm indexes sorted by mean values
        len_activeArms = len(self._activeArms)

        for i in range(self._nArms - 1):
            # Gaps Computations
            if i != 0:
                gap_l = self._meanArms[sorted_idx[i-1]] - self._meanArms[sorted_idx[i]]
                gap_l = round(gap_l, self._rounding)
            if i != len_activeArms - 1:
                gap_r = self._meanArms[sorted_idx[i]] - self._meanArms[sorted_idx[i+1]]
                gap_r = round(gap_r, self._rounding)

            # Checking if more Sampling is required based on Gaps
            current_cb = self._cb()
            if i == 0:
                if gap_r < current_cb and not self._JUMP_ARMORDERING:
                    c_print(1, "ORE.py, NOT ORDERED(): Gap_Right {} vs CB {}, arm 0 index: {}, activeArms {} sorted_idx {} means {}".format(gap_r, current_cb, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))
                    return True
            if i == len_activeArms - 1:
                if gap_l < current_cb and not self._JUMP_ARMORDERING:
                    c_print(1, "ORE.py, NOT ORDERED(): Gap_Left {} vs CB {}, last arm index: {}, activeArms {} sorted_idx {} means {}".format(gap_l, current_cb, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))
                    return True
            if i!= len_activeArms - 1 and i!= 0 and not self._JUMP_ARMORDERING:
                if gap_l < current_cb or gap_r < current_cb:
                    c_print(1, "ORE.py, NOT ORDERED(): Gap_Right {} Gap_Left {} vs CB {}, arm {} with Index: {}, activeArms {} sorted_idx {} means {}".format(gap_r, gap_l, current_cb, i, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))
                    return True

        # All arms are Separeted
        if not self._learnedPO:
            sorted_idx = argsort(self._meanArms)[::-1]
            c_print(4, "\n===LEARNED ARM ORDERING\nORE.py, ORDERED(): Learned partial order. Active arms {}, sorted {}, means {}".format(self._activeArms, sorted_idx, self._meanArms))
            # Variable setting for the next phase
            self._r = 1
            self._s = 0
            self._activeArms = sorted_idx

        return False


    def _discarded(self):
        for i in range(self._nArms - 1):
            len_activeArms = len(self._activeArms)

            # Due to the bucketing, #activeArms >= tau
            if len_activeArms == self._tau:
                c_print(1, "ORE.py, DISCARDING() TAU not ordered Arms. Active arms {}, Means {}, CB {}".format(self._activeArms, self._meanArms, self._cb()))
                return
            assert self._nArms - self._s == len_activeArms, "Inconsistent arm elimination"

            sorted_idx = argsort(self._meanArms)[::-1] # all indexes sorted desc by means   
            arm_deletion = False # set Canary

            # Gaps Computations
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
                    c_print(4, "\nORE.py, DISCARDING(): Gap_Right {} vs CB {}, arm 0 index: {}, activeArms {} sorted_idx {} means {}".format(gap_r, current_cb, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))
            if i == len_activeArms - 1:
                if gap_l > current_cb and sorted_idx[i] in self._activeArms: # Discarding the last index
                    arm_deletion = True
                    c_print(4, "\nORE.py, DISCARDING(): Gap_Left {} vs CB {}, last arm index: {}, activeArms {} sorted_idx {} means {}".format(gap_l, current_cb, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))
            if i!= len_activeArms - 1 and i!= 0:
                if gap_l > current_cb and gap_r > current_cb and sorted_idx[i] in self._activeArms: # Discarding an index in the middle
                    arm_deletion = True
                    c_print(4, "\nORE.py, DISCARDING(): Gap_Right {} Gap_Left {} vs CB {}, arm {} with Index: {}, activeArms {} sorted_idx {} means {}".format(gap_r, gap_l, current_cb, i, sorted_idx[i], self._activeArms, sorted_idx, self._meanArms))

            if arm_deletion: # Arm deletion based on one of previous cases
                idx = self._activeArms.remove(sorted_idx[i])
                self._s = self._s + 1
                return 
        return


    def overwriteArmMeans(self, means):
        assert self.LP == 2, "ORE.py, OVERWRITING ARM MEANS IN WRONG MOD"
        self._meanArms = means
        c_print(4, "Setting Arm Means {}".format(means))
        return

    def _bucketing(self, indexes):
        returned_list = []
        len_idx = len(indexes)
        n_full_chunks = int(len_idx/self._tau)
        bucketed = []

        if len(indexes) != self._tau:
            c_print(1, "Number of full chunks {}".format(n_full_chunks))
            # Bucketing the full chunks 
            for i in range(n_full_chunks):
                c_print(1, "Inserting in full chunks")
                l = min((len_idx, (i+1)*self._tau))
                returned_list += indexes[i*self._tau : l]
                returned_list += indexes[i*self._tau : l]

        # Fill the eventual last partial piece of the list with not pulled active arms
        if ceil(len(indexes)/self._tau) != int(len_idx/self._tau) or self._tau == len(indexes):
            c_print(1, "ORE.py, bucketing(), Completing the last chunk")
            remaining_indexes = indexes[n_full_chunks*self._tau : min(len_idx, (n_full_chunks+1)*self._tau)]
            c_print(1, "ORE.py, bucketing(), Remaining indexes {}".format(remaining_indexes))
            i = 0
            bucketed = remaining_indexes
            while len(bucketed) < self._tau:
                if self._activeArms[i] not in remaining_indexes:
                    bucketed.append(self._activeArms[i])
                    c_print(1, "Index i {}, bucketed {}".format(self._activeArms[i], bucketed))
                i = i + 1
            returned_list += bucketed + bucketed
        c_print(1, "ORE.py, bucketing(), Bucketed List: {}".format(returned_list))

        assert len(returned_list) == 2*self._tau*ceil(len_idx/self._tau), "Inconsistent bucketing"
        return returned_list
