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
houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']

medians = {}
counts = {}
means = {}

df = df.dropna(subset=['Hogwarts House'])

# compute medians
for feature in courses:
    medians[feature] = {}
    counts[feature] = {}
    means[feature] = {}
    for house in houses:
        try:
            house_data = df[feature][df['Hogwarts House'] == house]
            data = np.array(pd.to_numeric(
                house_data, errors='coerce').dropna().astype(float))
            # normalize data to improve readability of the graph
            data = sc().zscore(data)
        except KeyError as e:
            print(
                "Error: could not read features. Check that all features are present in the dataset.")
            print(f'Source: {e}')
            exit(1)
        # if we need to drop negative values
        # data = data[data >= 0]

        for i in data:
            if not isinstance(i, (int, float)):
                print(i)
                print("Error: invalid data")
                exit(1)
        medians[feature][house] = stats.median(data)
        counts[feature][house] = len(data)
        means[feature][house] = stats.mean(data)


# bar plot for each feature and show the graphs on same window

def plot_stat(stat, ylabel):
    plt.figure(figsize=(25, 15))

    Gryffindor = []
    Slytherin = []
    Ravenclaw = []
    Hufflepuff = []

    for feature in courses:
        Gryffindor.append(stat[feature]['Gryffindor'])
        Slytherin.append(stat[feature]['Slytherin'])
        Ravenclaw.append(stat[feature]['Ravenclaw'])
        Hufflepuff.append(stat[feature]['Hufflepuff'])
    
    X_axis = np.arange(len(courses))
    
    plt.bar(X_axis - 0.4, Gryffindor, 0.2, label = 'Gryffindor', color='red')
    plt.bar(X_axis - 0.2, Slytherin, 0.2, label = 'Slytherin', color='green')
    plt.bar(X_axis + 0.0, Ravenclaw, 0.2, label = 'Ravenclaw', color='dodgerblue')
    plt.bar(X_axis + 0.2, Hufflepuff, 0.2, label = 'Hufflepuff', color='goldenrod')
    
    plt.xticks(X_axis, courses, rotation=45)
    plt.xlabel("Courses")
    plt.ylabel(ylabel)
    plt.title("Score distribution between houses for each course")
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.legend()
    plt.show()

plot_stat(medians, "Median scores")
plot_stat(means, "Mean scores")
