#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:40:42 2017

@author: michael
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
import getopt


def avg_performance():
    data_points = []
    for problem_size in problem_sizes:
        bests = 0
        sys.stdout.write('\r'+problem_size+"      ")
        for seed in seeds:
            results_file = problem+'_'+problem_size+'_'+str(seed)+".csv"
            with open(results_folder+results_file, 'r') as f:
                data = pd.DataFrame.from_csv(f)
            bests += max(data['max'])
        avg_best = bests/(seed+1)
        data_points.append((problem_size, avg_best))

    x = [int(x[0]) for x in enumerate(data_points)]
    y = [float(x[1]) for x in data_points]
    print()
    return x, y


def num_optimals():
    data_points = []
    for problem_size in problem_sizes:
        optimals = 0
        sys.stdout.write('\r'+problem_size+"      ")
        for seed in seeds:
            results_file = problem+'_'+problem_size+'_'+str(seed)+".csv"
            with open(results_folder+results_file, 'r') as f:
                data = pd.DataFrame.from_csv(f)
            optimals += (max(data['max']) == 1)
        data_points.append((problem_size, optimals))

    x = [int(x[0]) for x in enumerate(data_points)]
    y = [float(x[1]) for x in data_points]
    print()
    return x, y


# Process commandline arguments here
opts, args = getopt.getopt(sys.argv[1:], 'hp:s:fo')
opts = dict(opts)

problem = opts['-p']
suffix = opts['-s'] if '-s' in opts else ''


plt.rcParams["font.family"] = "CMU Serif"
plt.rcParams['mathtext.fontset'] = 'cm'
plt.style.use('ggplot')
plt.figure(figsize=(16, 8))
ax = plt.axes(facecolor='w', frameon=True)
ax.grid(color='lightgray', linestyle='-', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')
ax.spines['right'].set_color('lightgray')


if 'mkp' in problem:
    problem_sizes = ['100_5_0.25', '100_5_0.5', '100_5_0.75', '100_10_0.25',
                     '100_10_0.5', '100_10_0.75', '100_30_0.25', '100_30_0.5',
                     '100_30_0.75', '250_5_0.25', '250_5_0.5', '250_5_0.75',
                     '250_10_0.25', '250_10_0.5', '250_10_0.75', '250_30_0.25',
                     '250_30_0.5', '250_30_0.75', '500_5_0.25', '500_5_0.5',
                     '500_5_0.75', '500_10_0.25', '500_10_0.5', '500_10_0.75',
                     '500_30_0.25', '500_30_0.5', '500_30_0.75']
    seeds = range(10)

else:
    problem_sizes = ['16', '25', '36', '49', '64', '81', '100', '121', '144',
                     '169', '196', '225', '256', '289', '324', '361', '400',
                     '441', '484', '529', '576', '625', '676', '729', '784']
    seeds = range(200)

algorithms = [
        '1+1EA',
        '1+1Fast-EA',
        '1+lambda,lambdaEA',
        '2+1GA',
        '5+1EA',
        '5+1GA', '20+20EA', '20+20GA', '20+20Fast-GA']

if problem == "mkp":
    algorithms.append('Chu-and-Beasley')
else:
    algorithms.append('P3-soft')

labels = {'1+1EA': r'(1+1) EA',
          '1+1Fast-EA': r'(1+1) Fast-EA',
          '1+lambda,lambdaEA': r'(1+($\lambda$, $\lambda$)) GA',
          '2+1GA': r'(2+1) GA',
          '5+1EA': r'(5+1) EA',
          '5+1GA': r'(5+1) GA',
          '20+20EA': r'(20+20) EA',
          '20+20GA': r'(20+20) GA',
          '20+20Fast-GA': r'(20+20) Fast-GA',
          'P3-soft': 'P3',
          'Chu-and-Beasley': 'Chu and Beasley'}
colours = {'1+1EA': 'blueviolet',
           '1+1Fast-EA': 'darkorange',
           '1+lambda,lambdaEA': 'deepskyblue',
           '2+1GA': 'green',
           '5+1EA': 'royalblue',
           '5+1GA': 'purple',
           '20+20EA': 'palevioletred',
           '20+20GA': 'indianred',
           '20+20Fast-GA': 'magenta',
           'P3-soft': 'saddlebrown',
           'Chu-and-Beasley': 'gray'}
markers = {'1+1EA': 'o',
           '1+1Fast-EA': 'o',
           '1+lambda,lambdaEA': 'x',
           '2+1GA': 'x',
           '5+1EA': 'o',
           '5+1GA': 'x',
           '20+20EA': 'o',
           '20+20GA': 'x',
           '20+20Fast-GA': 'x',
           'P3-soft': '^',
           'Chu-and-Beasley': 's'}

for algorithm in algorithms:
    print("\""+algorithm+"\",")
    results_folder = 'results/'+problem+suffix+'/'+algorithm+'/'
    print(results_folder)
    if '-f' in opts:
        x, y = avg_performance()
    elif '-o' in opts:
        x, y = num_optimals()

    plt.scatter(x, y, label=labels[algorithm], marker=markers[algorithm],
                c=colours[algorithm], clip_on=False)

if '-f' in opts:
    plt.ylabel("average fitness after 10000 fitness evaluations", fontsize=16)
    plt.ylim(ymax=1)
    savefile = "results/graphs/"+problem+suffix.replace("/", "-")+"-avg-performance.svg"
elif '-o' in opts:
    plt.ylabel("number of instances where optimal solution was found", fontsize=16)
    savefile = "results/graphs/"+problem+suffix.replace("/", "-")+"-optimal-solutions.svg"

plt.yticks(fontsize=16)
plt.xlim(-1, len(problem_sizes))
plt.xticks(range(len(problem_sizes)), problem_sizes, fontsize=14)

if problem == 'mkp':
    plt.xticks(range(len(problem_sizes)), problem_sizes, fontsize=14, rotation='vertical')
    plt.xlabel("problem instance", fontsize=16)
else:
    plt.xlabel("problem size", fontsize=16)

plt.tight_layout()

lgd = plt.legend(frameon=False, fontsize=14, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
lgd.get_frame().set_facecolor('w')


plt.savefig(savefile, bbox_extra_artists=(lgd,), bbox_inches='tight')
