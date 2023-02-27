#!/usr/bin/python3

import numpy as np
import pandas as pd
import os
import sys
path = os.path.join(os.path.dirname(__file__), '..', 'tools')
sys.path.insert(1, path)
from my_logistic_regression import MyLogisticRegression as MyLR


def main():
    # check args
    if len(sys.argv) != 2:
        print("Usage: python3 logreg_train.py dataset_train.csv")
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

    # check if the input file contains the houses
    df = df.dropna(subset=['Hogwarts House'])
    if len(df['Hogwarts House'].unique()) == 0:
        print("Error: empty file / file without houses")
        exit(1)

    # global parameters
    features = ['Astronomy',
                'Herbology',
                'Defense Against the Dark Arts',
                'Charms']
    # clean data
    df = df.dropna(subset=features)

    try:
        output_file = open('parameters.csv', 'w')
    except:
        print("Error: could not open output file")
        exit(1)

    # header of the csv file
    print("house,theta0,theta1,theta2,theta3,theta4", file=output_file)

    models = []
    for house in df['Hogwarts House'].unique():
        # create a model for each house (one-vs-all)
        thetas = np.array([0, 0, 0, 0, 0]).reshape(-1, 1)
        model = MyLR(thetas, alpha=5e-5, max_iter=100000)
        # train the model
        dataset = np.array(df[features]).reshape(-1, 4)
        house_or_not_list = []
        for i in df['Hogwarts House']:
            house_or_not_list += [1 if i == house else 0]
        house_or_not = np.array(house_or_not_list).reshape(-1, 1)
        model.fit_(dataset, house_or_not)
        # save the model
        models.append(model)
        # print the model
        print(f'{house},{model.theta[0][0]},{model.theta[1][0]},{model.theta[2][0]},'
              f'{model.theta[3][0]},{model.theta[4][0]}', file=output_file)

    output_file.close()
    

main()
