"""\
------------------------------------------------------------
USE: python <PROGNAME> (options) testfile
OPTIONS:
    -h            : print this help message
    -e EVALS      : the maximum number of fitness function evaluations before a
                    solution is presented, defaults to 10000
    -r SEED       : a random SEED to get the algorithm started,
                    defaults to 10000
------------------------------------------------------------
"""
import random
import os
import numpy as np

from statsmodels import robust

from deap import base
from deap import creator
from deap import tools

import ea_util as util

import sys
import getopt

# Process commandline arguments here
opts, args = getopt.getopt(sys.argv[1:], 'hp:fm:l:ce:b:S:s:a:r:')
opts = dict(opts)


def printHelp():
    help = __doc__.replace('<PROGNAME>', sys.argv[0], 1)
    print(help, file=sys.stderr)
    sys.exit()

##############################
# help option
if '-h' in opts:
    printHelp()

seed = opts['-r'] if '-r' in opts else 10000
max_evals = opts['-e'] if '-r' in opts else 10000
MU = 100  # MU is always 100 but clearer to leave it as a variable
# Lambda is always 1

# f = '../MKP/test_data2/MKP_100_5_3.txt'
# problem file
if len(args) < 1:
    print('ERROR: please provide a problem file')
    printHelp()

f = args[0]

if "\\" in f:
    print('ERROR: invalid file string. Please use / as a delimiter')
    printHelp()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("mate", tools.cxUniform, indpb=0.5)

toolbox.register("selectParents", tools.selTournament, tournsize=2)
toolbox.register("select", tools.selBest)


# returns one offspring as algorithm is designed to only create one new
# individual per generation
def crossoverAndMutation(population, toolbox):
    offspring = [toolbox.clone(ind) for ind in population]
    offspring = toolbox.mate(offspring[0], offspring[1])
    offspringM = toolbox.mutate(offspring[0])
    return offspringM[0]


def eaOne(population, toolbox, max_evals, MU, stats=None, halloffame=None):
    gen = 0
    # initialise logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # get initial fitnesses
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitness_count = len(invalid_ind)
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # initialise hall of fame
    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    # Begin the generational process
    # for gen in range(1, ngen+1):
    while(fitness_count < max_evals):
        parents = toolbox.selectParents(population, 2)

        # apply crossover and mutation to get one offspring
        offspring = crossoverAndMutation(parents, toolbox)
        # use repair operator on offspring
        offspring = util.repairKnapsack(offspring, capacities, coefficients,
                                        N, M)

        # if offspring is duplicate discard and go to next generation
        if any([np.array_equal(offspring, i) for i in population]):
            continue

        del offspring.fitness.values
        # evaluate fitness of offspring (can use much less code as only one
        # individual)

        fitness = toolbox.evaluate(offspring)
        offspring.fitness.values = fitness
        fitness_count += 1
        gen += 1

        # the child thing is necessary as deap is expecting a list of lists
        # (i.e. a pop) not just an individual
        child = [offspring]

        if halloffame is not None:
            halloffame.update(child)

        population = toolbox.select(population+child, MU)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=1, **record)

    return population, logbook


def main():
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = eaOne(pop, toolbox, max_evals, MU, stats, halloffame=hof)

    return pop, log, hof

if __name__ == "__main__":
    evals = []
    failures = 0
    random.seed(seed)
    summary = ''

    N, M, O, values, coefficients, capacities = util.readMKP(f)
    toolbox.register("individual", util.generateKnapsack,
                     creator.Individual, N, capacities, coefficients)
    toolbox.register("evaluate", util.evalKnapsack, O, values, capacities,
                     coefficients)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", tools.mutFlipBit, indpb=2/N)
    pop, log, hof = main()
    evals.append(sum(log.select("nevals")))
    # Set the output file here
    results_file = 'results/CB100Repair/'
    if not os.path.exists(results_file):
        os.makedirs(results_file)
    i = f.split('/')[-1]
    with open(results_file+i, 'w') as l:
            print(log, file=l)

    if (hof[0].fitness.values[0] < int(O)):
        failures += 1

    print("Run:", i,
          "Evals:", evals[-1],
          "MES:", np.median(evals),
          "MAD:", robust.mad(evals),
          "FAILURES:", failures,
          "Best:", hof[0].fitness.values[0],
          "Solution:", O)
