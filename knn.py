'''
Authors: Matthew Robinson, Antonio Munoz
Date: 11/23/20
Class: CSE 489 ML
Description: KNN Visualizer
Sources:
    - For color graph:      https://pythonspot.com/k-nearest-neighbors/
    - For helper functions: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
To-Do:
    - Add a new (black) point and dipslay k closest neighbors with distance
    - Interface with GUI
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


def main(num_classes, num_neighbors, num_points):
    # Number of data categories
    #num_classes = 3
    classes = []

    # Specifiying k
    #num_neighbors = 5

    # Randomly generate num_points up to max_value
    #num_points = 20
    max_value = 100
    coords = np.random.rand(num_points, 2) * max_value

    # Step size for graph
    h = 5

    # Array containing each class (0...n), where each element
    # corresponds to a color
    for i in range(0,num_classes):
        classes.append(i)

    # Extracting x and y coordinates
    x_coords = []
    y_coords = []
    for coord in coords:
        x_coords.append(coord[0])
        y_coords.append(coord[1])

    # Creating a points.csv file with our randomly generated points and categories
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
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # np.meshgrid creates a rectangular array from 2 1D arrays
    # np.arrange spaces an array by hj
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Create color maps
    if (num_classes == 2):
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    elif (num_classes == 3):
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00','#0011FF'])
    elif (num_classes == 4):
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF', '#FF8100'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00','#0011FF', '#FFCE00'])
    elif (num_classes == 5):
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF', '#FF8100', '#F000FF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00','#0011FF', '#FFCE00', '#FF00DB'])

    # We create an instance of Neighbors Classifier and fit the data.
    # Implements the k-nearest neighbors vote
    clf = neighbors.KNeighborsClassifier(num_neighbors, weights='distance')
    # Using the classifier to fit our coordinates with the target values (colors)
    clf.fit(coords, y)

    # Predict class labels
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])      # ravel() flattens an array
    Z = Z.reshape(xx.shape)

    # Setting up the plot
    plt.figure()
    # Create a color plot
    # xx and yy are coordinates of the corners, Z scales the data, and cmap is color map
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # Scattering our points
    plt.scatter(x_coords, y_coords, c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(str(num_classes) + "-Class classification (k = %i)" % (num_neighbors))
    plt.show()



if __name__=="__main__":
    main()