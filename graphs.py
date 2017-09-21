#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:40:42 2017

@author: michael
"""
import numpy as np
import pandas as pd
import os
import seaborn
import matplotlib.pyplot as plt
import sys

seaborn.set_context(rc={"lines.linewidth": 1.2})
seaborn.set_style("whitegrid")

plt.figure(figsize=(16, 8))
plt.rcParams["font.family"] = "CMU Serif"

problem = 'onemax'

if problem == 'mkp':
    problem_sizes = ['100_5_0.25', '100_5_0.5', '100_5_0.75', '100_10_0.25',
                     '100_10_0.5', '100_10_0.75', '100_30_0.25', '100_30_0.5',
                     '100_30_0.75', '250_5_0.25', '250_5_0.5', '250_5_0.75',
                     '250_10_0.25', '250_10_0.5', '250_10_0.75', '250_30_0.25',
                     '250_30_0.5', '250_30_0.75', '500_5_0.25', '500_5_0.5',
                     '500_5_0.75', '500_10_0.25', '500_10_0.5', '500_10_0.75',
                     '500_30_0.25', '500_30_0.5', '500_30_0.75']

else:
    problem_sizes = ['16', '25', '36', '49', '64', '81', '100', '121', '144',
                     '169', '196', '225', '256', '289', '324', '361', '400',
                     '441', '484', '529', '576', '625', '676', '729', '784']


for algorithm in os.listdir('results/'+problem):
    results_folder = 'results/'+problem+'/'+algorithm+'/'
    if not os.path.isdir(results_folder):
        continue
    print(results_folder)
    data_points = []
    bests = []
    for problem_size in problem_sizes:
        sys.stdout.write('\r'+problem_size)
        for results_file in os.listdir(results_folder):
            if (results_file.startswith(problem+'_'+problem_size+'_') and
                    results_file.endswith('.csv')):
                with open(results_folder+results_file) as f:
                    data = pd.DataFrame.from_csv(f)
                    bests.append(max(data['max']))

        avg_best = np.mean(bests)
        data_points.append((problem_size, avg_best))

        x = [int(x[0]) for x in enumerate(data_points)]
        y = [float(x[1]) for x in data_points]
    print()

    plt.scatter(x, y, label=algorithm, marker='x')

plt.title("Algorithm performance on the "+problem+" Problem",
          fontsize=18, y=1.04, color='black')
plt.xlabel("Problem Size", fontsize=16)
plt.ylabel("Average Fitness after 10000 fitness evaluations", fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(-1, len(problem_sizes))
plt.xticks(range(len(problem_sizes)), problem_sizes, fontsize=14)
if problem == 'mkp':
    plt.xticks(range(len(problem_sizes)), problem_sizes, fontsize=14, rotation='vertical')

plt.tight_layout()

plt.legend(frameon=True, fontsize=14, loc=3)
plt.savefig("results/"+problem+"/avg_performance.svg")
plt.show()
