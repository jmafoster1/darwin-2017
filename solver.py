"""\
------------------------------------------------------------
USE: python <PROGNAME> (options) <testfile>
Options with a * must be provided
OPTIONS:
    -h           : print this help message
    -p* PROBLEM  : the problem to be solved, one of either 'mkp', 'onemax',
                   'ising', or 'maxsat'
    -f           : solve the problem using a Fast EA (or GA if -c is provided)
    -m* MU       : maintain a population of MU individuals
                   (not required for '-a twoplusone')
    -l* LAMBDA   : generate LAMBDA children each generation
                   (not required for '-a twoplusone')
    -c           : apply crossover before mutation each generation
    -e EVALS     : the maximum number of fitness function evaluations before a
                   solution is presented, defaults to 10000
                   NOTE: This is not an exact cutoff and may go slightly over
                   depending on MU and LAMBDA values
    -b BETA      : the BETA value for used for Fast EAs and GAs
                   (defaults to 1.5)
    -S* SELECTION: the method of parent selection for crossover, one of either
                   'uniform', or 'tournament'
    -s SIZE      : the size of tournament for tournament selection
    -a* ALGORITHM: the algorithm to be used to solve the problem, one of either
                   'plus' for a (MU + LAMBDA) EA
                   'comma' for a (MU, LAMBDA) EA
                   'twoplusone' for a greedy (2 + 1) EA
                   'lambdalambda' for a 1 + (\lambda, \lambda) algorithm
    -r SEED      : a random SEED to get the algorithm started,
                   defaults to 100
    -A           : used for '-a lambdalambda' - whether or not to use the
                   self-adjusting variant of the algorithm - defaults to 1
    -F FLOAT     : used for '-a lambdalambda' - the F value for the
                   algorithm - defaults to 2.5
    -R           : turn on the repair operator for -p MKP

ARGUMENTS:
    For each execution, the name of a file containing the problem specification
    must be provided. When the '-p' option is 'mkp', 'ising', or 'maxsat', a
    text file must be provided. When the '-p' argument is 'onemax', it is
    sufficient to simply provide the following: 'onemax_$length_$run' where
    $length is the length of the bitstring solution and $run is the run number.

RESULTS:
    Results are saved to 'results/PROBLEM/$details/<testfile>
------------------------------------------------------------
"""

import random
import os
import numpy as np

from deap import base
from deap import creator
from deap import tools

import ea_util as util

import sys
import getopt

# Process commandline arguments here
opts, args = getopt.getopt(sys.argv[1:], 'hp:fm:l:ce:b:S:s:a:r:AF:R')
opts = dict(opts)


def printHelp():
    help = __doc__.replace('<PROGNAME>', sys.argv[0], 1)
    print(help, file=sys.stderr)
    sys.exit()


##############################
# help option
if '-h' in opts:
    printHelp()
##############################
mandatory = ['-p', '-S', '-a']
if any([i not in opts for i in mandatory]):
    print('ERROR: please provide all required options')
    printHelp()
if opts['-S'] == 'tournament' and '-s' not in opts:
    print('ERROR: please provide tournament size')
    printHelp()
if opts['-a'] == 'twoplusone' and any([i in opts for i in ['-l', '-m']]):
    print('ERROR: cannot provide -m or -l options for twoplusone algorithm')
    printHelp()
if opts['-a'] != 'twoplusone' and any([i not in opts for i in ['-l', '-m']]):
    print('ERROR: must provide -m and -l options unless using twoplusone')
    printHelp()
F = 2.5
if '-F' in opts:
    try:
        F = float(opts['-F'])
    except ValueError:
        print('ERROR: the -F option must be a number')
        printHelp()

util.setF(F)


if '-A' in opts:
    util.setAdjusting(1)
else:
    util.setAdjusting(0)

if '-R' in opts:
    util.setRepair(1)
else:
    util.setRepair(0)

# mandatory options
problem = opts['-p']
MU = 2 if opts['-a'] == 'twoplusone' else int(opts['-m'])
LAMBDA = 1 if opts['-a'] == 'twoplusone' else int(opts['-l'])
selection = opts['-S']
size = int(opts['-s']) if '-s' in opts else 1
algorithm = opts['-a']

# additional options
fast = '-f' in opts
crossover = '-c' in opts
max_evals = int(opts['-e']) if '-e' in opts else 10000
BETA = opts['-b'] if '-b' in opts else 1.5
seed = opts['-r'] if '-r' in opts else 100

# Need to set random seed in util file as well
random.seed(seed)
util.setSeed(seed)

# problem file
if len(args) < 1:
    print(('ERROR: please provide a problem file\n' +
           'This is also used as the results file name'))
    printHelp()
f = args[0]

algorithms = {'plus': util.plus,
              'comma': util.comma,
              'twoplusone': util.twoPlusOne,
              'lambdalambda': util.lambdalambda}
solvers = {'mkp': util.MKP, 'maxsat': util.maxSat, 'ising': util.ising,
           'onemax': util.oneMax}

creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selBest)

if selection == 'uniform':
    toolbox.register("selectParents", util.selParents, toolbox)
elif selection == 'tournament':
    toolbox.register("selectParents", tools.selTournament, tournsize=size)

# Set the output folder here
results_folder = ('results/' + problem + '/' +
                  ('Greedy ' if algorithm == 'twoplusone' else '') +
                  str(MU) + '+' + str(LAMBDA) +
                  ((', '+ str(LAMBDA)) if algorithm == 'lambdalambda' else '') +
                  ('Fast ' if fast else '') +
                  ('GA' if crossover else 'EA') + '/')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

solvers[problem](MU, LAMBDA, algorithms[algorithm], fast, crossover,
                 results_folder, toolbox, f, max_evals, BETA)