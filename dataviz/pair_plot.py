#!/usr/bin/python3


import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
path = os.path.join(os.path.dirname(__file__), '..', 'tools')
sys.path.insert(1, path)

# check args
if len(sys.argv) != 2:
    print("Usage: python3 pair_plot.py dataset.csv")
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
features = ['Arithmancy',
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
            'Flying',
            'Best Hand',
            'Birthday']

df = df.dropna(subset=['Hogwarts House'])
# transform best hand to 0 or 1
df['Best Hand'] = df['Best Hand'].map({'Left': 0, 'Right': 1})
# transform birth date to age
df['Birthday'] = df['Birthday'].map(lambda x: 2020 - int(x.split('-')[2]))


def plot_pair():
    try:
        sns.pairplot(df, hue='Hogwarts House', vars=features)
        plt.show()
    except Exception as e:
        print(f'Error: {e}')
        exit(1)
plot_pair()

# Best features are:
    # Astronomy
    # Herbology
    # Defense Against the Dark Arts
    # Charms
