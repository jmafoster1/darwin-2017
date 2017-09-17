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
                   (not required for '-a greedy' or '-a lambdalambda')
    -l* LAMBDA   : generate LAMBDA children each generation
                   (not required for '-a greedy')
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
                   'greedy' for a greedy (2 + 1) EA
                   'lambdalambda' for a 1 + (\lambda, \lambda) algorithm
    -r SEED      : a random SEED to get the algorithm started,
                   defaults to 100
    -A           : used for '-a lambdalambda' - whether or not to use the
                   self-adjusting variant of the algorithm - defaults to 1
    -F FLOAT     : used for '-a lambdalambda' - the F value for the
                   algorithm - defaults to 2.5
    -R           : turn on the repair operator for -p mkp

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


algorithms = {'plus': util.theory_GA,
              'comma': util.theory_GA,
              'greedy': util.greedy,
              'lambdalambda': util.lambdalambda}
solvers = {'mkp': util.MKP, 'maxsat': util.maxSat, 'ising': util.ising,
           'onemax': util.oneMax}

options = {'F': 1.5}


##############################
# help option
if '-h' in opts:
    printHelp()
##############################
mandatory = ['-p', '-S', '-a']
if any([i not in opts for i in mandatory]):
    print('ERROR: please provide all required options')
    printHelp()
if opts['-a'] not in algorithms.keys():
    print('ERROR: invalid algorithm, -a option must be one of',
          list(algorithms.keys()))
    printHelp()
if opts['-p'] not in solvers.keys():
    print('ERROR: invalid problem, -p option must be one of',
          list(solvers.keys()))
if opts['-S'] == 'tournament' and '-s' not in opts:
    print('ERROR: please provide tournament size')
    printHelp()
if opts['-a'] == 'greedy' and '-l' in opts:
    print('ERROR: greedy (MU + 1) algorithm can only produces one child ' +
          '- invalid option -m')
    printHelp()
if opts['-a'] == 'greedy' and '-m' not in opts:
    print('ERROR: must provide -m option for greedy (MU + 1) algorithm ' +
          '- invalid option -m')
    printHelp()
if (opts['-a'] not in ['greedy', 'lambdalambda'] and
   any([i not in opts for i in ['-l', '-m']])):
    print('ERROR: must provide -m and -l options unless using greedy or ' +
          'lambdalambda')
    printHelp()
if '-F' in opts:
    try:
        options['F'] = float(opts['-F'])
    except ValueError:
        print('ERROR: the -F option must be a number')
        printHelp()
if '-R' in opts and opts['p'] != 'mkp':
    print('ERROR: -R option only relevant for -p mkp option')
    printHelp()
if '-f' in opts and opts['-a'] not in ['plus', 'comma']:
    print('ERROR: the -f option is only valid for -a plus and -a comma')
    printHelp()
if opts['-a'] == 'lambdalambda' and ('-m' in opts or '-l' in opts):
    print('ERROR: cannot provide -m or -l options for -a lambdalambda')
    printHelp()

options['adjusting'] = '-A' in opts
options['repair'] = '-R' in opts

# mandatory options
options['problem'] = opts['-p']
try:
    options['mu'] = int(opts['-m']) if opts['-a'] != 'lambdalambda' else 1
except ValueError:
    print('ERROR: the -m option must be an integer')
    printHelp()

try:
    if opts['-a'] == 'lambdalambda':
        options['lambda'] = 'lambda'
    elif opts['-a'] == 'greedy':
        options['lambda'] = 1
    else:
        options['lambda'] = int(opts['-l'])
    
except ValueError:
    print('ERROR: the -l option must be an integer')
    printHelp()

options['selection'] = opts['-S']
if '-s' in opts:
    try:
        options['tournsize'] = int(opts['-s'])
    except ValueError:
        print('ERROR: the -s option must be an integer')
        printHelp()
options['algorithm'] = opts['-a']

# additional options
options['fast'] = '-f' in opts
options['crossover'] = '-c' in opts
options['max_evals'] = 10000
if '-e' in opts:
    try:
        options['max_evals'] = int(opts['-e'])
    except ValueError:
        print('ERROR: the -e option must be an integer')
        printHelp()
options['beta'] = 1.5
if '-b' in opts:
    try:
        options['F'] = float(opts['-b'])
    except ValueError:
        print('ERROR: the -b option must be a number')
        printHelp()
options['seed'] = 100
if '-r' in opts:
    try:
        options['seed'] = int(opts['-r'])
    except ValueError:
        print('ERROR: the -r option must be an integer')
        printHelp()

# Need to set random seed in util file as well
random.seed(options['seed'])

# problem file
if len(args) < 1:
    print(('ERROR: please provide a problem file\n' +
           'This is also used as the results file name'))
    printHelp()

# Set the output folder here
results_folder = ('results/' + options['problem'] + '/' +
                  ('Greedy ' if options['algorithm'] == 'greedy' else '') +
                  str(options['mu']) + '+' + str(options['lambda']) +
                  ((', ' + str(options['lambda'])) if options['algorithm'] == 'lambdalambda' else '') +
                  ('Fast ' if options['fast'] else '') +
                  ('GA' if options['crossover'] else 'EA') + '/')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

options['algorithm_fn'] = algorithms[options['algorithm']]
options['results_folder'] = results_folder
options['problem_file'] = args[0]
options['solver'] = solvers[options['problem']]
util.main(options)
