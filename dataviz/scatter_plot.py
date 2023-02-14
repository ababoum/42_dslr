#!/usr/bin/python3


import pprint
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
path = os.path.join(os.path.dirname(__file__), '..', 'tools')
sys.path.insert(1, path)
from Scaler import Scaler as sc
from TinyStatistician import TinyStatistician as ts

# check args
if len(sys.argv) != 2:
    print("Usage: python3 scatter_plot.py dataset.csv")
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


df = df.dropna(subset=['Hogwarts House'])


def plot_scatter(excl_list=[]):
    plt.figure(figsize=(25, 15))
    plt.gcf().subplots_adjust(bottom=0.2)
    display_list = [feature for feature in courses if feature not in excl_list]
    index = 0
    for feature in courses:
        if feature in excl_list:
            continue
        try:
            house_data = df[feature]
            data = np.array(pd.to_numeric(
                house_data, errors='coerce').dropna().astype(float))
            plt.scatter(index * np.ones((len(data), 1)), data)
            index += 1
        except KeyError as e:
            print(
                "Error: could not read features. Check that all features are present in the dataset.")
            print(f'Source: {e}')
            exit(1)

    plt.xticks(range(len(display_list)), display_list, rotation=45)
    plt.title('All Houses')
    plt.xlabel('Courses')
    plt.ylabel('Scores')
    plt.show()


exclude = []
plot_scatter(exclude)
exclude.append('Arithmancy')
plot_scatter(exclude)
exclude += ['Astronomy', 'Muggle Studies'] # good candidates
plot_scatter(exclude)
exclude += ['Ancient Runes', 'Transfiguration', 'Flying']
plot_scatter(exclude)
exclude += ['Charms', 'Care of Magical Creatures']
plot_scatter(exclude)