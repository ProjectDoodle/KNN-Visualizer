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
import csv

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
    #num_classes = 3
    classes = ['Blue', 'Green', 'Red']
    num_neighbors = 5
    # Step size for graph
    h = 5
    # Randomly generate num_points up to max_value
    num_points = 20
    max_value = 100
    coords = np.random.rand(num_points, 2) * max_value

    # Extracting x and y coordinates
    x_coords = []
    y_coords = []
    for coord in coords:
        x_coords.append(coord[0])
        y_coords.append(coord[1])

    # Creating a points.csv file with our randomly generated points
    with open('points.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['X', 'Y', 'Color'])
        for coord in coords:
            filewriter.writerow([coord[0], coord[1], random.choice(classes)])

    # Assigning the newly created csv file to a variable
    points = pd.read_csv('points.csv')
    y = points['Color'].values

    # Calculating min, max, and limits
    x_min, x_max = min(x_coords) - 1, max(x_coords) + 1
    y_min, y_max = min(y_coords) - 1, max(y_coords) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])

    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(num_neighbors, weights='distance')
    # X is training data and y is targets
    clf.fit(coords, y)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')


    plt.scatter(x_coords, y_coords, c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)" % (num_neighbors))
    plt.show()



if __name__=="__main__":
    main()