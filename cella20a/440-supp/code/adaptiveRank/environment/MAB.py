''' Environment for a multi-armed bandit problem'''

__version__ = "0.1"

from adaptiveRank.environment.Environment import Environment
from adaptiveRank.Results import *
from adaptiveRank.tools.io import c_print
from adaptiveRank.arm.Bernoulli import Bernoulli
from adaptiveRank.arm.HashBernoulli import HashBernoulli

from numpy import arange, argmax, around, array, linspace, unique, where, zeros
from random import seed, randint
import sortednp as snp

class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""

    def __init__(self, horizon, nbBuckets, gamma, fraTop, maxDelay, binary_rewards, modality, policy_name, SWITCHING, SC):
        c_print(4, "MAB.py, INIT horizon {}, nbBuckets {}, gamma {}, fraTop {}, maxDelay {}, binary_rewards {}, policy {}, switching {}".format(horizon, nbBuckets, gamma, fraTop, maxDelay, binary_rewards,  policy_name, SWITCHING))
        self.horizon = horizon
        self.nbBuckets = nbBuckets
        self.gamma = gamma
        self.fraTop = fraTop
        self.maxDelay = maxDelay
        self._binary_rewards = binary_rewards # Specifies whether to use binary rewards or not
        self._modality = modality # Specifies the learning problem: 0 full, 1 arm ordering, 2 rank estimation
        self.policy_name = policy_name
        self.arms = [] # List of Bernoulli/HashBernoulli objects
        self.nbArms = 0
        self.r_star = 0

        ### Switching Costs
        self._SC = SC
        ### Test parameter
        self._SWITCHING = SWITCHING # Canary for given means


    def compute_states(self):
        '''Called at every step in the play() method. Manages the trajectory evolution.'''
        assert len(self._armsDelay) == len(self._armsStates), "MAB compute_states: Incoherent size"

        for i, arm, delay in zip(self._armsIndexes, self.arms, self._armsDelay):
            if self._SWITCHING == True and delay > self.maxDelay:
                self._armsStates[i] = arm.computeState(0)
            else:
                self._armsStates[i] = arm.computeState(int(delay))
            c_print(1, "MAB.py, states() Index {}, arm {}, delay {}".format(i, arm, delay))

        c_print(1, "MAB.py, states() {}".format(self._armsStates))
        return


    def play(self, policy, horizon, nbRepetition):
        ''' Called once per policy from __init__ of Evaluation. Rounds scheduler.'''
        c_print(1, "MAB.py, play()")

        # Result data structure initialization
        result = Result(horizon)
        t = 0

        # Arm Creation
        self.nbArms = self._arm_creation(nbRepetition)
        self._armsDelay = [0]*self.nbArms
        self._armsIndexes = arange(self.nbArms)
        assert self._armsIndexes[-1] == self.nbArms -1, "Wrong arm creation"
        c_print(4, "MAB.py, play(), init: Arm Indexes {} Binary Rewards {}".format(self._armsIndexes, self._binary_rewards))
        self._armsStates = zeros(self.nbArms) # Expected rewards of each arm according to suffered delays
        self.r_star = self._r_star_computation()

        result.setNbArms(self.nbArms)

        # Learning Modality Message Passing
        if self._modality == 2 and self.policy_name in ['PI Low', 'PI ucb']: # Rank Estimation 
            policy.overwriteArmMeans(self._meanArms)

        while t < horizon:
            c_print(1, "\n===\nMAB.py, play(): round {}\n===".format(t))

            if t == 0:
                self.compute_states()
                c_print(1, "MAB.py, play(): current delays: {}".format(self._armsDelay))
                c_print(1, "MAB.py, play(): arm states: {}".format(self._armsStates))
                # RStar policy update
                if self.policy_name == "Ghost":
                    policy.initialize(self.r_star)

            # Structured Choice and Feedback 
            choice = policy.choice(self._armsStates)
            current_len = len(set(choice))

            # Cost computation
            cost = 0.0
            if self._SC and t != 0 and self._past_len != current_len:
                c_print(1, "Past len {} current len {}".format(self._past_len, current_len))
                cost = 1.0

            tmp = 0
            for c in choice:
                tmp = tmp + 1
                if self._SWITCHING == True and self._armsDelay[c] > self.maxDelay:
                    expected_reward, reward = self.arms[c].draw(0, nbRepetition)
                else:
                    expected_reward, reward = self.arms[c].draw(self._armsDelay[c], nbRepetition)

                c_print(1, "\nMAB.py, play(): Chosen arm: {} at round: {} with rwd {}".format(c, t, reward))
                c_print(1, "MAB.py, play(): Arm {}".format(self.arms[c]))
                c_print(1, "MAB.py, play(): arm states: {}".format(self._armsStates))
                c_print(1, "MAB.py, play(): Suffered delays: {}".format(self._armsDelay))

                policy.update(c, reward, self._armsDelay[c])

                # Reward with (possible) cost penalization
                if self._SC == 1:
                    if self._binary_rewards:
                        result.store(t, c, (1.0*reward) - cost)
                    else:
                        result.store(t, c, (1.0*expected_reward) - cost)
                    cost = 0.0
                else:
                    if self._binary_rewards:
                        result.store(t, c, reward)
                    else:
                        result.store(t, c, expected_reward)

                # Delays update
                for i in self._armsIndexes:
                    d = int(self._armsDelay[i])
                    # Not the chosen arm and already pulled once
                    if d != 0 and i != c:
                        self._armsDelay[i] = int(self._armsDelay[i]) + 1
                    # I cannot put it to zero or it seems like an unpulled arm
                    if d > self.maxDelay:
                        self._armsDelay[i] = int(self.maxDelay) + 1
                    if i == c:
                        self._armsDelay[i] = 1

                # States update
                self.compute_states()

                # Additional termination condition due to finite horizon
                if t == horizon - 1:
                    return result
                t = t + 1
                self._past_len = len(set(choice))

        return result


    def _arm_creation(self, seed_init):
        if self._SWITCHING == False:
            self._meanArms = []
            seed(seed_init)
            starting_grid = linspace(0.0, 1.0, self.nbBuckets, endpoint = True)
            c_print(4, "MAB.py arm_creation() Buckets: {}".format(starting_grid))
            delta = 1.0/(self.nbBuckets) # Previously adopted delta
            new_extreme = delta*(self.fraTop)*self.nbBuckets
            good_arms = linspace(0, new_extreme, self.nbBuckets, endpoint = False)
            c_print(4, "MAB.py arm_creation() Good arms: {} with extreme point: {}".format(good_arms, new_extreme))
            means = array(snp.merge(starting_grid, good_arms)) # evenly round to 2 decimals 
            means = around(unique(means), 3)
            self._meanArms = 1 - means
            nbArms = len(means)
            c_print(4, "\n=========MAB_INIT=========")
            c_print(4, "MAB.py, arm_creation(), Arm means: {}".format(self._meanArms))
            for i in range(nbArms):
                mu = self._meanArms[i]
                #if self._SWITCHING == False:
                delayLB = 1
                delayUB = 1 + randint(seed_init, self.maxDelay - 1)
                gamma = self.gamma
                #else:
                #    delayLB = self._given_delaysLB[i]
                #    delayUB = self._given_delaysUB[i]
                #    gamma = self._given_gamma
                tmpArm = Bernoulli(mu, gamma, delayLB, delayUB, self._binary_rewards)
                c_print(1, "MAB.py, arm_creation(), Created arm: {}".format(tmpArm))
                self.arms.append(tmpArm)
        else:
            #1
            self.arms.append(HashBernoulli(0.7, 0.75, 1.0, self._binary_rewards))
            self.arms.append(HashBernoulli(0.606, 0.65, 0.86, self._binary_rewards))
            #2
            #self.arms.append(HashBernoulli(0.65, 0.6, 1.0, self._binary_rewards))
            #self.arms.append(HashBernoulli(0.55, 0.5, 0.84, self._binary_rewards))
            #3
            #self.arms.append(HashBernoulli(0.5, 0.55, 1.0, self._binary_rewards))
            #self.arms.append(HashBernoulli(0.4, 0.45, 0.81, self._binary_rewards))
            self.maxDelay = 2
            self._meanArms = [1.0, 0.8]
            nbArms = len(self._meanArms)
        return nbArms


    def _r_star_computation(self):
        avgs = [self._avg(i) for i in  np.arange(1,len(self.arms) + 1,1)]
        max_element = array(avgs).max()
        bin_mask = where(array(avgs)==max_element)[0]
        r_star = max(bin_mask)
        c_print(4, "MAB.py, r_star_comp(), Obtained avgs: {}, max_el {}, bin_mask {}, r_star: {}".format(avgs, max_element, bin_mask, r_star))
        return r_star


    def _avg(self, r):
        c_print(1, "\nMAB.py, _avg(), Computing rank {}'s average".format(r))
        delayed_means = [arm.computeState(r) for arm in self.arms[:r]]
        c_print(1, "MAB.py, _avg(), First {} arm means: {}".format(r, delayed_means[:r]))
        partial_sum = sum(delayed_means[:r])
        avg = around(partial_sum / r, 3)
        c_print(1, "MAB.py, _avg(), Partial Sum {:f}, Average {:f}. over {} arms".format(partial_sum, avg, r))
        return avg

