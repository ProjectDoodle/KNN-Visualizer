'''
Authors: Matthew Robinson, Antonio Munoz
Date: 11/23/20
Class: CSE 489 ML
Description: KNN Visualizer
Sources:
    - For color graph:      https://pythonspot.com/k-nearest-neighbors/
    - For helper functions: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
'''



# Standard powerful python tools
#import matplotlib
#matplotlib.use('GTK3Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Used for splitting the data
from sklearn.model_selection import train_test_split
# Scalar preprocesser used to deal with bias
from sklearn.preprocessing import StandardScaler
# Classifier we're using
from sklearn.neighbors import KNeighborsClassifier
# Testing metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
#from numpy import random


'''
Description: Computes the euclidean distance
Param x: x coordinate point
Param y: y coordinate
'''
def euclidean_distance(x, y):
    distance = 0.0
    for i in range(len(x) - 1):
        distance += (x[i] - y[i]) ** 2
    return math.sqrt(distance)


'''
Description: Finds the nearest neighbors
Param train:
Param test_row:
Param num_neighbors:
'''
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors



def main():
    # Step size
    h = 0.2
    # Randomly generate num_points up to max_value
    num_points = 20
    max_value = 50
    coords = np.random.rand(num_points, 2) * max_value

    # Extracting x and y coordinates
    x_coords = []
    y_coords = []
    for coord in coords:
        x_coords.append(coord[0])
        y_coords.append(coord[1])


    # Calculating min, max, and limits
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    print(xx)
    print(yy)


    plt.figure()
    plt.scatter(x_coords,y_coords)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Data points")
    plt.show()



if __name__=="__main__":
    main()