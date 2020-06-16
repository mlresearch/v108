__version__ = "0.1"

from adaptiveRank.tools.io import c_print
from adaptiveRank.arm import Bernoulli
from adaptiveRank.Evaluation import Evaluation
from adaptiveRank.environment import MAB
from adaptiveRank.policies.UCB import UCB
from adaptiveRank.policies.Ghost import Ghost
from adaptiveRank.policies.RStar import RStar
from adaptiveRank.policies.Greedy import Greedy
from adaptiveRank.policies.FPO_UCB import FPO_UCB
from adaptiveRank.policies.Ore import ORE2
from adaptiveRank.policies.One import One
from adaptiveRank.policies.Alt import Alt

from math import sqrt
from optparse import OptionParser
from numpy import mean, std, zeros, arange, where
from joblib import Parallel, delayed
import matplotlib
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size':'15.0'})
rc('text', usetex=True)
import matplotlib.pyplot as plt
import sys
sys.maxsize = 1000000 # Avoid truncations in print

#====================
# RUNNING PARAMETERS
#====================
parser = OptionParser(usage="usage: %prog [options]",
        version="%prog 1.0")
parser.add_option('--gamma', dest = 'GAMMA', default = 0.999, type = "float", help = "Discount parameter")
parser.add_option('--max_delay', dest = 'MAX_DELAY', default = 6, type = "int", help = "Memory size for the discount")
parser.add_option('--tau', dest = 'TAU', default = '7', type = 'int', help = 'Sampling delay')
parser.add_option('-T', dest = 'T', default = 500000, type = "int", help = "Time horizon")
parser.add_option('-k', dest = 'N_BUCKETS', default = 8, type = "int", help = "Number of buckets")
parser.add_option('--fra_top', dest = 'FRA_TOP', default = 0.2, type = "float", help = "Fraction of top arms")
#parser.add_option('--delay_ub', dest = 'DELAY_UB', default = 2, type = "int", help = "Gap from the delay bar")
parser.add_option('--delta', dest = "DELTA", default = 0.1, type = "float", help = "confidence in estimates")
parser.add_option('--n_rep', dest = 'N_REP', default = 1, type = "int", help = "Number of repetitions")
parser.add_option('--rounding', dest = 'ROUNDING', default = 5, type = "int", help = "Number of kept decimals")
parser.add_option('--bin', dest = 'BINARY', default = 1, type = "int", help = "Binary rewards")
parser.add_option('-v', dest = 'VERBOSE', default = '1', type = 'int', help = "Verbose in terms of plots")
parser.add_option('-s', dest = 'STORE', default = '1', type = 'int', help = "Storing plots")
parser.add_option('--stage', dest = 'MOD', default = '0', type = 'int', help = "0 - full learning, 1 arm ordering, 2 rank estimation")
parser.add_option('--switch', dest = 'SWITCH', default = '0', type = 'int', help = "Testing with specified means")
parser.add_option('--sc', dest = 'SC', default = '0', type = 'int', help = "With switching costs")
(opts, args) = parser.parse_args()

# Parsing parameters
GAMMA = opts.GAMMA
MAX_DELAY = opts.MAX_DELAY
DELTA = opts.DELTA
TAU = opts.TAU
FRA_TOP = opts.FRA_TOP
HORIZON = opts.T
N_BUCKETS = opts.N_BUCKETS
N_REPETITIONS = opts.N_REP
ROUNDING = opts.ROUNDING # number of kept decimals
VERBOSE = opts.VERBOSE
STORE = opts.STORE
BINARY = opts.BINARY # Binary rewards
MOD = opts.MOD # Running modality
SWITCHING = opts.SWITCH # Given arms
SC = opts.SC # With switching costs

#=====================
# INITIALIZATION 
#===================== 
if MOD != 1: # Useless benchmarks for the arm ordering estimation problem
    policies = [RStar(HORIZON, 2)]#, UCB(HORIZON, 2)]
    policies_name = ['Ghost']#, 'UCB1']
else:
    policies = []
    policies_name = []

# Appending an additional benchmark to study the policy non-stationarity effect
#if TEST == True:
#    policies.append(One())
#    policies_name.append('PI stationary')
#    policies.append(Alt())
#    policies_name.append('PI alternating')
#else:
policies.append(FPO_UCB(HORIZON, TAU, DELTA, ROUNDING, 5, BINARY, MOD, 1))
policies_name.append('PI ucb')
shrink = 1
policies.append(ORE2(HORIZON, TAU, DELTA, shrink, ROUNDING, 5, BINARY, MOD))
policies_name.append('PI Low')

assert len(policies) == len(policies_name), "Check consistency of policy naming"
N_POLICIES = len(policies_name)
cumSumRwd = zeros((N_POLICIES, N_REPETITIONS, HORIZON))

#=====================
# RUN OVER POLICIES
#=====================
results = []
for i,p in enumerate(policies):
    mab = MAB(HORIZON, N_BUCKETS, GAMMA, FRA_TOP, MAX_DELAY, BINARY, MOD, policies_name[i], SWITCHING, SC)
    c_print(5, "=========RUN_POLICIES=========")
    c_print(5, "===Run.py, Run {}/{}. Policy: {}".format(i,len(policies_name)-1, policies_name[i]))
    evaluation = Evaluation(mab, p, HORIZON, policies_name[i], N_REPETITIONS)
    results.append(evaluation.getResults())
    nbArms = evaluation.getNbArms()

#=====================
# PLOTTING RESULTS
#=====================
if opts.VERBOSE:
    COLORS = ['b', 'g', 'r', 'y', 'k', 'c', 'm']
    MARKERS = ['o', '+', 'x', 'v', 'o', '+', 'x']
    POLICY_AVGS = []
    POLICY_STD = []
    POLICY_N = []
    plt_fn =  plt.plot
    fig = plt.figure(1)
    #plt.title("Gamma: {}, Max Delay: {}, Arms: {}, Fraction Top-Arms: {}".format(GAMMA, MAX_DELAY, nbArms, FRA_TOP))
    ax = fig.add_subplot(1,1,1)
    i = 0
    for name,avg,std in results:
        POLICY_AVGS.append(avg)
        POLICY_STD.append(std)
        POLICY_N.append(name)
        plt.fill_between(arange(HORIZON), avg - (std/2), avg + (std/2), alpha = 0.5, color = COLORS[i])
        lbl = '$\displaystyle\pi_{ghost}$'
        if name == 'PI Low':
            lbl = '$\displaystyle\pi_{low}$'
        if name == 'PI ucb':
            lbl = '$\displaystyle\pi_{ucb}$'
        plt_fn(arange(HORIZON), avg, color = COLORS[i], marker = MARKERS[i], markevery=HORIZON/100, label = lbl)
        i+=1
    plt.legend(loc=2)
    #plt.xlabel('Rounds')
    #plt.ylabel('Cumulative Reward')
    plt.grid()
    if STORE > 0:
        prefix = ['full', 'arm_ordering', 'rank_estimation']
        if SC:
            plt.savefig("output/{}_SC_g{}_ft{}_d{}_dUB{}_dBar{}_bin{}_T{}_k{}.png".format(prefix[MOD], GAMMA, FRA_TOP, DELTA, MAX_DELAY, TAU, BINARY, HORIZON, N_BUCKETS))
        else:
            plt.savefig("output/{}_g{}_ft{}_d{}_dUB{}_dBar{}_bin{}_T{}_k{}.png".format(prefix[MOD], GAMMA, FRA_TOP, DELTA, MAX_DELAY, TAU, BINARY, HORIZON, N_BUCKETS))

        if MOD != 1: # Not Arm Ordering
            ghost_index = POLICY_N.index('Ghost')
            piucb_index = POLICY_N.index('PI ucb')
            pilow_index = POLICY_N.index('PI Low')

            avg_regret_ucb = POLICY_AVGS[ghost_index] - POLICY_AVGS[piucb_index]
            avg_regret_low = POLICY_AVGS[ghost_index] - POLICY_AVGS[pilow_index]

            std_regret_ucb = POLICY_STD[ghost_index] + POLICY_STD[piucb_index]
            std_regret_low = POLICY_STD[ghost_index] + POLICY_STD[pilow_index]


            # Regret figure creation
            plt_fn = plt.plot
            fig = plt.figure()
            #plt.title("Regret wrt Ghost, Gamma {}, Max Delay {}, Fra. Top-Arms {}".format(GAMMA, MAX_DELAY, FRA_TOP))
            ax = fig.add_subplot(1,1,1)

            # PI UCB plot
            plt.fill_between(arange(HORIZON), avg_regret_ucb - (std_regret_ucb/2), avg_regret_ucb + (std_regret_ucb/2), alpha = 0.5, color = COLORS[piucb_index])
            plt_fn(arange(HORIZON), avg_regret_ucb, color = COLORS[piucb_index], marker = MARKERS[piucb_index], markevery=HORIZON/100, label = '$\displaystyle\pi_{ucb}$')

            # PI low plot
            plt.fill_between(arange(HORIZON), avg_regret_low - (std_regret_low/2), avg_regret_low + (std_regret_low/2), alpha = 0.5, color = COLORS[pilow_index])
            plt_fn(arange(HORIZON), avg_regret_low, color = COLORS[pilow_index], marker = MARKERS[pilow_index], markevery=HORIZON/100, label = '$\displaystyle\pi_{low}$')


            if SWITCHING and not SC:
                plt.legend(loc=3) #upper left
            else:
                plt.legend(loc=2) #lower left
            #plt.xlabel('Rounds')
            #plt.ylabel('Regret')
            #plt.tight_layout()
            plt.autoscale(enable = True, axis ='x', tight=True)
            plt.grid()
            ax.set_xscale('log')

            if SC:
                if SWITCHING:
                    plt.savefig("output/regret_SWITCHING_SC_bin{}_T{}.png".format(BINARY, HORIZON))
                else:
                    plt.savefig("output/regret_SC_g{}_ft{}_d{}_dUB{}_dBar{}_bin{}_T{}_k{}.png".format(GAMMA, FRA_TOP, DELTA, MAX_DELAY, TAU, BINARY, HORIZON, N_BUCKETS))
            else:
                if SWITCHING:
                    plt.savefig("output/regret_SWITCHING_NOSC_bin{}_T{}".format(BINARY, HORIZON))
                else:
                    plt.savefig("output/regret_NOSC_g{}_ft{}_d{}_dUB{}_dBar{}_bin{}_T{}_k{}.png".format(GAMMA, FRA_TOP, DELTA, MAX_DELAY, TAU, BINARY, HORIZON, N_BUCKETS))
