#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:18:55 2017

@author: michael
"""

import pandas as pd
import os
import sys
import shutil

problem = 'mkp'
for algorithm in os.listdir('results/'+problem):
    results_folder = 'results/'+problem+'/'+algorithm+'/'
    if not os.path.isdir(results_folder):
        continue
    print(results_folder)
#        for results_file in os.listdir(results_folder):
#            sys.stdout.write('\r'+problem+': '+algorithm+': '+results_file+'                  ')
#            if results_file.endswith('.txt'):
#                with open(results_folder+results_file) as f:
#                    columns = [x.strip() for x in f.readline().split("\t")]
#                    rows = []
#                    for line in f:
#                        rows.append([float(x.strip()) for x in line.split("\t")])
##                    
#                data = pd.DataFrame(data=rows, columns=columns)
#                with open(results_folder+results_file.replace('.txt', '.csv'), 'w') as f:
#                    data.to_csv(path_or_buf=f)
#                os.remove(results_folder+results_file)
    shutil.make_archive(results_folder, 'zip', results_folder)

