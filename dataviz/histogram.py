#!/usr/bin/python3


import pprint
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
path = os.path.join(os.path.dirname(__file__), '..', 'tools')
sys.path.insert(1, path)
from TinyStatistician import TinyStatistician as ts
from Scaler import Scaler as sc

# check args
if len(sys.argv) != 2:
    print("Usage: python3 histogram.py dataset_train.csv")
    exit(1)

# read csv
try:
    df = pd.read_csv(sys.argv[1])
except:
    print("Error: could not read file")
    exit(1)

# check if empty
if df.empty:
    print("Error: empty file")
    exit(1)

# global parameters
stats = ts()
courses = ['Arithmancy',
           'Astronomy',
           'Herbology',
           'Defense Against the Dark Arts',
           'Divination',
           'Muggle Studies',
           'Ancient Runes',
           'History of Magic',
           'Transfiguration',
           'Potions',
           'Care of Magical Creatures',
           'Charms',
           'Flying']

### PLOT ALL HISTOGRAMS ON THE SAME WINDOW, DIFFERENT FIGURES ###

fig = plt.figure()
fig.set_size_inches(15, 25)
gs = fig.add_gridspec(13, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Courses histograms')
for i, feature in enumerate(courses):
    try:   
        data = df[feature].dropna()
        # Normalize data
        data = sc.zscore(np.array(data.values))
        axs[i].hist(data, bins=10, label=feature, alpha=0.5)
        axs[i].legend()
    except KeyError as e:
        print(
            "Error: could not read features. Check that all features are present in the dataset.")
        print(f'Source: {e}')
        exit(1)
plt.show()

### PLOT ALL HISTOGRAMS ON THE SAME WINDOW, SAME FIGURE ###
plt.figure(figsize=(15, 25))
for feature in courses:
    try:   
        data = df[feature].dropna()
        # Normalize data
        plt.title('Courses histograms')
        data = sc.zscore(np.array(data.values))
        plt.hist(data, bins=10, label=feature, alpha=0.5)
        plt.legend()
    except KeyError as e:
        print(
            "Error: could not read features. Check that all features are present in the dataset.")
        print(f'Source: {e}')
        exit(1)
plt.show()