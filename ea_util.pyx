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
from statsmodels import robust
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
    for i in range(len(clauses)):
        for c in range(3):
            if (individual[clauses[i][c]] == signs[i][c]):
                total += 1
                break
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


def lambdalambda(population, toolbox, MU, LAMBDA, N, crossover,
                 stats, hof, fast, max_evals, BETA):
    x = toolbox.individual()
    x.fitness.values = toolbox.evaluate(x)

    LAMBDA = 1

    gen = 0
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitness_count = len(invalid_ind)
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    # Begin the generational process
    while(fitness_count < max_evals and fitnessValue(hof[0]) < 1):
        gen += 1
        k = LAMBDA
        c = 1.0/k
        p = LAMBDA/N

        toolbox.register("mate", tools.cxUniform, indpb=c)

        # Mutation phase
        ell = np.random.binomial(N, p)
        X = [mut_l(toolbox.clone(x), ell) for i in range(LAMBDA)]
        x_prime = max(X, key=lambda i: fitnessValue(i))
        fitness_count += len(X)

        # Crossover phase
        Y = [cross_c(toolbox.clone(x), toolbox.clone(x_prime), toolbox) for i in range(LAMBDA)]
        y = max(Y, key=lambda i: fitnessValue(i))

        if fitnessValue(y) > fitnessValue(x):
            x = y
            LAMBDA = max([(LAMBDA/F), 1])
        elif fitnessValue(y) == fitnessValue(x):
            x = y
            LAMBDA = min([(LAMBDA*F**0.25), N])
        elif fitnessValue(y) < fitnessValue(x):
            LAMBDA = min([(LAMBDA*F**0.25), N])

        LAMBDA = int(LAMBDA)

        population = [y]
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        #  Update the hall of fame with the generated individuals
        if hof is not None:
            hof.update([x])

    return population, logbook


def theory_GA(population, toolbox, int MU, int LAMBDA, int N, crossover, stats,
          hof, fast, int max_evals, float BETA, selection):
    gen = 0

    # initialise logbook
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # initialise individuals fitness
    eval_count = len(population)
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # initialise hall of fame
    if hof is not None:
        hof.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)

    # Begin the generational process
    while(eval_count < max_evals and fitnessValue(hof[0]) < 1):
        gen += 1
        if(fast):
            alpha = fmut(N, BETA)
            toolbox.register("mutate", tools.mutFlipBit, indpb=alpha/N)

        #  Generate offspring
        offspring = []

        # if crossover is being used it is done before mutation
        if crossover:
            for i in range(LAMBDA):
                p1, p2 = toolbox.selectParents(population, 2)
                toolbox.mate(p1, p2)
                offspring += [p1]
        else:
            offspring = [toolbox.selectParents(population, 1)[0] for i in range(LAMBDA)]

        for off in offspring:
            off, = toolbox.mutate(off)
            del off.fitness.values

        #  Evaluate the individuals with an invalid fitness
        eval_count += len(offspring)
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if hof is not None:
            hof.update(offspring)

        # Select the next generation, favouring the offspring in the event
        # of equal fitness values
        if selection == 'plus':
            population = favourOffspring(population, offspring, MU)
        elif selection == 'comma':
            population = toolbox.select(offspring, MU)


        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)

    return population, logbook


def plus(population, toolbox, int MU, int LAMBDA, int N, crossover, stats,
          hof, fast, int max_evals, float BETA):
    return theory_GA(population, toolbox, MU, LAMBDA, N, crossover, stats,
          hof, fast, max_evals, BETA, 'plus')


def comma(population, toolbox, int MU, int LAMBDA, int N, crossover, stats,
          hof, fast, int max_evals, float BETA):
    return theory_GA(population, toolbox, MU, LAMBDA, N, crossover, stats,
          hof, fast, max_evals, BETA, 'comma')


def favourDiversity(population, MU):
    counts = []
    for i in population:
        count = 0
        for j in population:
            if list(i) == list(j):
                count += 1
        counts.append((i, len(population)/count))
    counts.sort(key=lambda x: (fitnessValue(x[0]), x[1]), reverse=True)
    return [i for i, _ in counts][:MU]


def twoPlusOne(population, toolbox, int MU, int LAMBDA, int N, crossover,
               stats, hof, fast, int max_evals, float BETA):
#    toolbox.register("mutate", tools.mutFlipBit, indpb=(1+math.sqrt(5))/(2*N))
    gen = 0
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitness_count = len(invalid_ind)
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    # Begin the generational process
    while(fitness_count < max_evals and fitnessValue(hof[0]) < 1):
        gen += 1
        p1, p2 = [toolbox.clone(ind) for ind in population]
        offspring = None
        if (fitnessValue(p1) > fitnessValue(p2)):
            offspring = p1
        elif (fitnessValue(p2) > fitnessValue(p1)):
            offspring = p2
        else:
            #p1, p2 = toolbox.selectParents(population, 2)
            toolbox.mate(p1, p2)
            offspring = p1
        offspring, = toolbox.mutate(offspring)
        del offspring.fitness.values
        offspring.fitness.values = toolbox.evaluate(offspring)

        offList = [offspring]
        # Update the hall of fame with the generated individuals
        if hof is not None:
            hof.update(offList)

#        # same fitness so tie break
#        if (all([fitnessValue(x) == fitnessValue(offspring) for x in population])):
#            if (list(population[0]) != list(offspring) and
#                list(population[1]) != list(offspring)):
#                population[random.randint(0, 1)] = offspring
#                
#        # different fitness so just select best
#        else:
#            population = toolbox.select(population+offList, MU)
        population = favourDiversity(population+offList, MU)

        #  Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        fitness_count += len(invalid_ind)

    return population, logbook


def main(N, MU, LAMBDA, BETA, t_box, crossover, fast, algorithm, max_evals):
    global toolbox
    toolbox = t_box
    population = toolbox.population(n=MU)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    toolbox.register("mutate", tools.mutFlipBit, indpb=1/N)

    pop, log = algorithm(population, toolbox, MU, LAMBDA, N, crossover, stats,
                         hof, fast, max_evals, BETA)

    return pop, log, hof


def printResults(hof, log, results_folder, f):
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


def oneMax(MU, LAMBDA, algorithm, fast, crossover, results_folder, toolbox, f,
           max_evals, BETA):
    N = int(f.split('_')[1])

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, N)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)

    # Run the algorithm
    pop, log, hof = main(N, MU, LAMBDA, BETA, toolbox, crossover, fast, algorithm, max_evals)

    printResults(hof, log, results_folder, f)


def ising(MU, LAMBDA, algorithm, fast, crossover, results_folder, toolbox, f,
          max_evals, BETA):
    N = int(f.split('_')[2])

    (min_energy, solution, number_of_spins,
     spins) = readIsing(f)
    span = number_of_spins - min_energy

    toolbox.register("evaluate", evalIsing, spins, min_energy, span)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, N)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)

    # Run the algorithm
    pop, log, hof = main(N, MU, LAMBDA, BETA, toolbox, crossover, fast, algorithm, max_evals)

    printResults(hof, log, results_folder, f)


def maxSat(MU, LAMBDA, algorithm, crossover, fast, results_folder, toolbox, f,
           max_evals, BETA):
    N = int(os.path.basename(f).split('_')[1])
    solution, signs, clauses = readMaxSat(f)

    # Structure initializers
    toolbox.register('evaluate', evalMaxSat, signs, clauses)
    toolbox.register('individual', tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, N)
    toolbox.register('population', tools.initRepeat, list,
                     toolbox.individual)

    pop, log, hof = main(N, MU, LAMBDA, BETA, toolbox, crossover, fast, algorithm, max_evals)

    printResults(hof, log, results_folder, f)


def MKP(MU, LAMBDA, algorithm, fast, crossover, results_folder, toolbox, f,
        max_evals, BETA):
    N, M, O, values, coefficients, capacities = readMKP(f)
    
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
    pop, log, hof = main(N, MU, LAMBDA, BETA, toolbox, crossover, fast, algorithm, max_evals)
    printResults(hof, log, results_folder, f)
