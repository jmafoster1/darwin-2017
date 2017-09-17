import random
import math
import os

from scipy.optimize import minimize

from deap import base
from deap import tools
from deap import creator

import numpy as np
import operator as op

from numpy.random import choice
from functools import reduce

cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t
ctypedef np.float_t DTYPE_f

global precision
precision = 65536
global F
F = 2.5
global adjusting
adjusting = 1
global repair
repair = 0
global toolbox
global ordering
global hof
global stats


def setRepair(int r):
    global repair
    repair = r


def setF(float f):
    global F
    F = f


def setAdjusting(int a):
    global adjusting
    adjusting = a


def setSeed(seed):
    random.seed(seed)


def log(logbook, population, gen, nevals):
    record = stats.compile(population) if stats else {}
    logbook.record(gen=gen, nevals=nevals, **record)
    if hof is not None:
        hof.update(population)


def fitnessValue(individual):
    return individual.fitness.values[0]


def powerDistribution(int n, float BETA):
    return reduce((lambda x, y: x + (y ** -BETA)), range(1, round(n/2)))


def favourOffspring(parents, offspring, MU):
    choice = (list(zip(parents, [0]*len(parents))) +
              list(zip(offspring, [1]*len(offspring))))
    choice.sort(key=lambda x: (fitnessValue(x[0]), x[1]), reverse=True)
    return [x[0] for x in choice[:MU]]
    

def fmut(int N, float BETA):
    CB = powerDistribution(N, BETA)
    alphas = list(range(1, int(N/2)))
    probs = [((CB ** -1) * (alphas[i] ** -BETA)) for i in range(len(alphas))]
    draw = choice(alphas, 1, p=probs)
    return draw


def selectParents(toolbox, individuals, k):
    parents = [random.choice(individuals) for i in range(k)]
    return [toolbox.clone(ind) for ind in parents]


def float_round(float value, int precision):
    return round(value * precision) / precision


def evalOneMax(np.ndarray[DTYPE_t, ndim=1] individual):
    return float(np.sum(individual)/individual.size),


def readIsing(file):
    with open(file, 'r') as f:
        min_energy, solution = f.readline().split(' ')
        min_energy = int(min_energy)
        solution = [int(x) for x in solution.strip()]
        number_of_spins = int(f.readline())
        spins = []
        for line in f:
            spins.append([int(x) for x in line.split(' ')])
        spins = np.array(spins)
        solution = np.array(solution)
        return min_energy, solution, number_of_spins, spins


def evalIsing(np.ndarray[DTYPE_t, ndim=2] spins, int min_energy, int span,
              np.ndarray[DTYPE_t, ndim=1] individual):
    cdef np.ndarray[DTYPE_t, ndim = 1] bit_to_sign = np.array([-1, 1])
    energy = - sum([(bit_to_sign[individual[spin[0]]] *
                     spin[2] *
                     bit_to_sign[individual[spin[1]]]) for spin in spins])
    return float_round(1.0 - (energy - min_energy) / span, precision),


def evalMaxSat(signs, clauses, individual):
    total = 0
    for (c1, c2, c3), (s1, s2, s3) in zip(clauses, signs):
            if (individual[c1] == s1):
                total += 1
            elif (individual[c2] == s2):
                total += 1
            elif (individual[c3] == s3):
                total += 1
    return float_round(float(total) / len(clauses), precision),


def readMaxSat(filename):
    with open(filename) as f:
        solution = [int(x) for x in f.readline().strip()]
        clauses = []
        signs = []
        for line in f:
            clause = []
            sign = []
            for pair in line.strip().split(' '):
                s, v = [int(x) for x in pair.split(',')]
                clause.append(v)
                sign.append(s)
            clauses.append(clause)
            signs.append(sign)
        return solution, signs, clauses


def evalOneMax(np.ndarray[DTYPE_t, ndim=1] individual):
    return float(np.sum(individual)/individual.size),


def readMKP(filename):
    with open(filename) as f:
        N, M, O = [float(x) for x in f.readline().split()]
        values = [float(x) for x in f.readline().split()]
        coefficients = []
        for line in f:
            if len(line.split()) == 0:
                break
            else:
                coefficients.append([float(x) for x in line.split()])
        capacities = [float(x) for x in f.readline().split()]
        return (int(N), int(M), O, np.array(values),
                np.array(coefficients), np.array(capacities))


def generateKnapsack(container, int size,
                     np.ndarray[DTYPE_f, ndim=1] capacities,
                     np.ndarray[DTYPE_f, ndim=2] coefficients):
    # Start with an empty bag
    ind = np.zeros(size, dtype=int)
    # Add items to the bag until it's not valid
    while validKnapsack(ind, capacities, coefficients):
        index = random.randint(0, size-1)
        ind[index] = 1
    # Remove the last item so we've got a valid bag again
    ind[index] = 0
    return container(ind)


def validKnapsack(np.ndarray[DTYPE_t, ndim=1] individual,
                  np.ndarray[DTYPE_f, ndim=1] capacities,
                  np.ndarray[DTYPE_f, ndim=2] coefficients):
    constraints = np.dot(coefficients, individual)
    return all([i < 0 for i in(constraints - capacities)])


def lp_relaxed(weights,
               values,
               float capacity,
               int N):
    cons = ({'type': 'ineq', 'fun': lambda x: np.dot(x, weights) - capacity})
    bnds = [(0, 1) for x in range(N)]
    return minimize(lambda x: np.dot(x, values),  # objective function
                   np.zeros(N),                   # initial guess
                   method='SLSQP',
                   bounds=bnds,
                   constraints=cons).fun


def repairKnapsack(np.ndarray[DTYPE_f, ndim=1] capacities,
                   np.ndarray[DTYPE_f, ndim=1] values,
                   np.ndarray[DTYPE_f, ndim=2] coefficients,
                   int N, int M, individual):
    for j in ordering:
        if individual[j] and not validKnapsack(individual, capacities, coefficients):
            individual[j] = 0
    for j in reversed(ordering):
        if not individual[j]:
            individual[j] = 1
        if not validKnapsack(individual, capacities, coefficients):
            individual[j] = 0
    return individual


def evalKnapsack(int M,
                 int N,
                 float O,
                 np.ndarray[DTYPE_f, ndim=1] values,
                 np.ndarray[DTYPE_f, ndim=1] capacities,
                 np.ndarray[DTYPE_f, ndim=2] coefficients,
                 np.ndarray[DTYPE_t, ndim=1] individual):
    if repair == 1:
        individual = repairKnapsack(capacities, values, coefficients, N, M, individual)
    if validKnapsack(individual, capacities, coefficients):
        return float(np.dot(individual, values)/O),
    return 0,


def mut_l(individual, l):
    bits = np.random.choice(list(range(len(individual))), l, replace=False)
    for bit in bits:
        individual[bit] = not individual[bit]
    individual.fitness.values = toolbox.evaluate(individual)
    return individual


def cross_c(p1, p2, toolbox):
    toolbox.mate(p1, p2)
    p1.fitness.values = toolbox.evaluate(p1)
    return p1


def bestFitness(population):
    return fitnessValue(tools.selBest(population, 1)[0])


def lambdalambda(options):
    # initialise logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    x = toolbox.individual()
    x.fitness.values = toolbox.evaluate(x)
    population= [x]
    LAMBDA = 1
    gen = 0
    eval_count = 1

    log(logbook, population, gen, len(population))

    # Begin the generational process
    while(eval_count < options['max_evals'] and bestFitness(population) < 1):
        nevals = 0
        gen += 1
        k = LAMBDA
        c = 1.0/k
        p = LAMBDA/options['N']

        toolbox.register("mate", tools.cxUniform, indpb=c)

        # Mutation phase
        ell = np.random.binomial(options['N'], p)
        X = [mut_l(toolbox.clone(x), ell) for i in range(LAMBDA)]
        x_prime = max(X, key=lambda i: fitnessValue(i))
        nevals += len(X)

        # Crossover phase
        Y = [cross_c(toolbox.clone(x), toolbox.clone(x_prime), toolbox) for i in range(LAMBDA)]
        y = max(Y, key=lambda i: fitnessValue(i))
        nevals += len(Y)

        if fitnessValue(y) > fitnessValue(x):
            x = y
            LAMBDA = max([(LAMBDA/F), 1])
        elif fitnessValue(y) == fitnessValue(x):
            x = y
            LAMBDA = min([(LAMBDA*F**0.25), options['N']])
        elif fitnessValue(y) < fitnessValue(x):
            LAMBDA = min([(LAMBDA*F**0.25), options['N']])

        LAMBDA = int(LAMBDA)
        eval_count += nevals

        population = [y]
        log(logbook, population, gen, nevals)

    return population, logbook


def theory_GA(options):
    # initialise logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # initialise individuals fitness
    population = toolbox.population(n=options['mu'])
    eval_count = len(population)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    gen = 0
    log(logbook, population, gen, len(population))

    # Begin the generational process
    while(eval_count < options['max_evals'] and bestFitness(population) < 1):
        gen += 1
        if(options['fast']):
            alpha = fmut(options['N'], options['beta'])
            toolbox.register("mutate", tools.mutFlipBit, indpb=alpha/options['N'])

        #  Generate offspring
        offspring = []

        # if crossover is being used it is done before mutation
        if options['crossover']:
            for i in range(options['lambda']):
                p1, p2 = toolbox.selectParents(population, 2)
                toolbox.mate(p1, p2)
                offspring += [p1]
        else:
            offspring = [toolbox.selectParents(population, 1)[0] for i in range(options['lambda'])]

        for off in offspring:
            off, = toolbox.mutate(off)

        #  Evaluate the individuals with an invalid fitness
        eval_count += len(offspring)

        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        # Select the next generation, favouring the offspring in the event
        # of equal fitness values
        if options['algorithm'] == 'comma':
            population = toolbox.select(offspring, options['mu'])
        else:
            population = favourOffspring(population, offspring, options['mu'])

        log(logbook, population, gen, len(offspring))

    return population, logbook


def greedy(options):
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.618/options['N'])
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    gen = 0
    population = toolbox.population(n=options['mu'])
    eval_count = len(population)

    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    
    log(logbook, population, gen, len(population))

    # Begin the generational process
    while(eval_count < options['max_evals'] and bestFitness(population) < 1):
        gen += 1
        maxFit = max([fitnessValue(x) for x in population])
        bestIndividuals = [x for x in population if fitnessValue(x) == maxFit]
        y1 = toolbox.clone(random.choice(bestIndividuals))
        y2 = toolbox.clone(random.choice(bestIndividuals))
        toolbox.mate(y1, y2)
        y_prime = y1
        y_prime, = toolbox.mutate(y_prime)
        y_prime.fitness.values = toolbox.evaluate(y_prime)
        eval_count += 1
        population.sort(key=lambda x: fitnessValue(x))
        z = population[0]  # population sorted worst to best
        if (fitnessValue(y_prime) >= fitnessValue(z) and
            all([list(x) != list(y_prime) for x in population])):
            population[0] = y_prime


        log(logbook, population, gen, 1)

    return population, logbook


def main(options):
    creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    global toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("select", tools.selBest)
    
    if options['selection'] == 'uniform':
        toolbox.register("selectParents", selectParents, toolbox)
    elif options['selection'] == 'tournament':
        toolbox.register("selectParents", tools.selTournament, tournsize=options['tournsize'])
    
    global hof
    hof = tools.HallOfFame(1, similar=np.array_equal)
    global stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    options['solver'](options)
    
    toolbox.register("mutate", tools.mutFlipBit, indpb=1/options['N'])

    pop, log = options['algorithm_fn'](options)
    printResults(log, options['results_folder'], options['problem_file'])

    return pop, log


def printResults(log, results_folder, f):
    f = f.split('/')[-1]
    with open(results_folder+f, 'w') as l:
        l.write(str(log))

    # Print result
    result = ("Run:" + f +
              " Evals:" + str(sum(log.select("nevals"))) +
#              " MES:" + str(np.median(evals)) +
#              " MAD:" + str(robust.mad(evals)) +
#              " FAILURES:" + str(failures) +
              " Best:" + str(fitnessValue(hof[0])) +
              " Solution:1.0")
    print(result)


def oneMax(options):
    options['N'] = int(os.path.basename(options['problem_file']).split('_')[1])

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, options['N'])
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)


def ising(options):
    min_energy, solution, number_of_spins, spins = readIsing(options['problem_file'])
    options['N'] = len(solution)
    span = number_of_spins - min_energy

    toolbox.register("evaluate", evalIsing, spins, min_energy, span)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, options['N'])
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)


def maxSat(options):
    solution, signs, clauses = readMaxSat(options['problem_file'])
    options['N'] = len(solution)

    # Structure initializers
    toolbox.register('evaluate', evalMaxSat, signs, clauses)
    toolbox.register('individual', tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, options['N'])
    toolbox.register('population', tools.initRepeat, list,
                     toolbox.individual)

def MKP(options):
    N, M, O, values, coefficients, capacities = readMKP(options['problem_file'])
    options['N'] = N
    
    Omega = [lp_relaxed(weights, values, capacity, N) for weights, capacity in zip(coefficients, capacities)]
    U = [values[j]/sum([Omega[i] * coefficients[i][j] for i in range(M)]) for j in range(N)]
    global ordering
    ordering = [x[0] for x in sorted(list(enumerate(U)), key=lambda x: x[1])]
    
    toolbox.register("evaluate", evalKnapsack, M, N, O, values, capacities,
                     coefficients)
    toolbox.register("individual", generateKnapsack,
                     creator.Individual, N, capacities, coefficients)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
