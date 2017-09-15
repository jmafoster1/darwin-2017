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


def binomialMutate(N, p):
    return np.random.binomial(N, p)


def fitnessValue(individual):
    return individual.fitness.values[0]


def powerDistribution(int n, float BETA):
    return reduce((lambda x, y: x + (y ** -BETA)), range(1, round(n/2)))


def fmut(int N, float BETA):
    CB = powerDistribution(N, BETA)
    alphas = list(range(1, int(N/2)))
    probs = [((CB ** -1) * (alphas[i] ** -BETA)) for i in range(len(alphas))]
    draw = choice(alphas, 1, p=probs)
    return draw


def selParents(toolbox, individuals, k):
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

def llOffSpring(population, toolbox, LAMBDA, N):
        offspring = []  # Copy old generators
        binom = binomialMutate(N, LAMBDA/N)
        numbers = list(range(0, N-1))
        for i in range(round(LAMBDA)):
            offspring.append(toolbox.clone(population[0]))
            list100 = np.random.choice(numbers, binom)
            for x in range(len(list100)):
                index = list100[x]
                offspring[i][index] = not offspring[i][index]
            del offspring[i].fitness.values
        nevals = len(offspring)
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        bestOne = toolbox.select(offspring, 1)[0]
        for j in range(round(LAMBDA)):
            bestOneClone = toolbox.clone(bestOne)
            parentClone = toolbox.clone(population[0])
            toolbox.mateBest(parentClone, bestOneClone)
            offspring[j] = parentClone
            del offspring[j].fitness.values
        return offspring, nevals


def lambdalambda(population, toolbox, int MU, int LAMBDA, int N, crossover,
                 stats, hof, fast, int max_evals, float BETA):   
    toolbox.register("mateBest", tools.cxUniform, indpb=1.0/LAMBDA)

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
    while(fitness_count < max_evals):
        parent = toolbox.clone(population[0])
        offspring, nevals = llOffSpring(population, toolbox, LAMBDA, N)
        fitness_count += nevals
        gen += 1

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        fitness_count += len(invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        #  Update the hall of fame with the generated individuals
        if hof is not None:
            hof.update(offspring)
        population = toolbox.select(population+offspring, MU)
        if adjusting != 0:
            if(parent.fitness.values[0] >= population[0].fitness.values[0]):
                LAMBDA = min(N, (LAMBDA * (math.pow(F, 0.25))))
            else:
                LAMBDA = round(max(1, (float(LAMBDA)/float(F))))

            toolbox.register("mateBest", tools.cxUniform, indpb=1.0/LAMBDA)
            toolbox.register("mutate", tools.mutFlipBit, indpb=(float(LAMBDA)/N))

        #  Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind) + nevals, **record)

    return population, logbook


def plus(population, toolbox, int MU, int LAMBDA, int N, crossover, stats,
          hof, fast, int max_evals, float BETA):
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
    while(eval_count < max_evals):
        gen += 1
        if(fast):
            alpha = fmut(N, BETA)
            toolbox.register("mutate", tools.mutFlipBit, indpb=alpha/N)

        #  Generate offspring
        offspring = []

        # if crossover is being used it is done before mutation
        if crossover:
            p1, p2 = toolbox.selectParents(population, 2)
            for i in range(math.ceil(LAMBDA/2.0)):
                toolbox.mate(p1, p2)
                offspring += [p1, p2]
        else:
            offspring = [toolbox.selectParents(population, 1)[0] for i in range(LAMBDA)]
        
        offspring = offspring[:LAMBDA]

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

        population = toolbox.select(population+offspring, MU)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)

    return population, logbook


def comma(population, toolbox, int MU, int LAMBDA, int N, crossover, stats,
          hof, fast, int max_evals, float BETA):
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
    while(eval_count < max_evals):
        gen += 1
        if(fast):
            alpha = fmut(N, BETA)
            toolbox.register("mutate", tools.mutFlipBit, indpb=alpha/N)

        #  Generate offspring
        offspring = []

        # if crossover is being used it is done before mutation
        if crossover:
            p1, p2 = toolbox.selectParents(population, 2)
            for i in range(math.ceil(LAMBDA/2.0)):
                toolbox.mate(p1, p2)
                offspring += [p1, p2]
        else:
            offspring = [toolbox.selectParents(population, 1)[0] for i in range(LAMBDA)]
        
        offspring = offspring[:LAMBDA]

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

        population = toolbox.select(offspring, MU)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)

    return population, logbook



def twoPlusOne(population, toolbox, int MU, int LAMBDA, int N, crossover,
               stats, hof, fast, int max_evals, float BETA):
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
    while(fitness_count < max_evals):
        gen += 1
        if (fitnessValue(population[0]) == fitnessValue(population[1])):
            offspring = toolbox.clone(population[random.randint(0, 1)])
            offspring2 = toolbox.clone(population[random.randint(0, 1)])
            offspring, offspring2 = toolbox.mate(offspring, offspring2)
            offspring, = toolbox.mutate(offspring)
            del offspring.fitness.values
            offspring.fitness.values = toolbox.evaluate(offspring)
        elif (fitnessValue(population[0]) > fitnessValue(population[1])):
            offspring = toolbox.clone(population[0])
            offspring, = toolbox.mutate(offspring)
            del offspring.fitness.values
            offspring.fitness.values = toolbox.evaluate(offspring)
        else:
            offspring = toolbox.clone(population[1])
            offspring, = toolbox.mutate(offspring)
            del offspring.fitness.values
            offspring.fitness.values = toolbox.evaluate(offspring)

        offList = [offspring]
        # Update the hall of fame with the generated individuals
        if hof is not None:
            hof.update(offList)

        # same fitness so tie break
        if (fitnessValue(population[0]) == fitnessValue(population[1]) and
                fitnessValue(population[0]) == fitnessValue(offList[0])):
            # all identical so it doesnt matter which 2
            if (list(population[0]) == list(population[1]) and
                    list(population[0]) == list(offList[0])):
                # population doesn't change
                break
            # offspring is unique
            elif (list(population[0]) == list(population[1])):
                population[1] = offList[0]
        # different fitness so just select best
        else:
            population = toolbox.select(population+offList, MU)

        #  Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if max(logbook.select("max")) == 1.0:
            break

        fitness_count += len(invalid_ind)

    return population, logbook


def main(N, MU, LAMBDA, BETA, toolbox, crossover, fast, algorithm, max_evals):
    population = toolbox.population(n=MU)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

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
#    print(result)


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
    toolbox.register("mutate", tools.mutFlipBit, indpb=1/N)
    pop, log, hof = main(N, MU, LAMBDA, BETA, toolbox, crossover, fast, algorithm, max_evals)
    printResults(hof, log, results_folder, f)
