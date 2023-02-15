#!/usr/bin/python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
path = os.path.join(os.path.dirname(__file__), '..', 'tools')
sys.path.insert(1, path)
from my_logistic_regression import MyLogisticRegression as MyLR


# global parameters
features = ['Astronomy',
            'Herbology',
            'Defense Against the Dark Arts',
            'Charms']


def main():
    # check args
    if len(sys.argv) != 3:
        print("Usage: python3 pair_plot.py dataset_test.csv parameters.csv")
        exit(1)

    # read dataset csv
    try:
        df = pd.read_csv(sys.argv[1])
    except:
        print("Error: could not read dataset file")
        exit(1)
    # check if empty
    if df.empty:
        print("Error: empty file")
        exit(1)

    # read parameters csv
    try:
        parameters = pd.read_csv(sys.argv[2])
    except:
        print("Error: could not read parameters file")
        exit(1)
    # check if empty
    if parameters.empty:
        print("Error: empty file")
        exit(1)

    # check parameters file
    try:
        if len(parameters['house'].unique()) != 4:
            raise Exception()
    except:
        print("Error: wrong format parameters file")
        exit(1)


    # create output file
    try:
        output_file = open("houses.csv", "w")
    except:
        print("Error: could not create output file")
        exit(1)

    print("Index,Hogwarts House", file=output_file)

    # prepare the models
    models = []
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    for house in houses:
        thetas = np.array(
            parameters[parameters['house'] == house].iloc[0][1:]
            ).astype(np.float64).reshape(-1, 1)
        model = MyLR(thetas)
        models.append(model)

    # predict the houses
    dataset = np.array(df[features]).reshape(-1, 4)
    for i in range(dataset.shape[0]):
        # predict the house for each student
        max_proba = 0
        max_proba_index = 0
        for j in range(len(models)):
            line = dataset[i].reshape(1, -1)
            proba = models[j].predict_(line)
            if proba > max_proba:
                max_proba = proba
                max_proba_index = j
        print(f'{i},{houses[max_proba_index]}', file=output_file)
        # add the house to the dataset
        df.loc[i, 'Hogwarts House'] = houses[max_proba_index]

    output_file.close()

    # plot the results
    try:
        sns.pairplot(df, hue='Hogwarts House', vars=features)
        plt.show()
    except Exception as e:
        print(f'Error: {e}')
        exit(1) 
        


main()
