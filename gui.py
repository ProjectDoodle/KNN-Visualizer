'''
Authors: Matthew Robinson, Antonio Munoz
Date: 11/23/20
Class: CSE 489 ML
Description: KNN Visualizer
To-Do:
    - Add a new (black) point and dipslay k closest neighbors with distance on drawn lines
    - Redo plot to have more control over coordinates (or figure out how with current implementation)
Notes:
    - I believe the current plot setup should work (if we can update with an added point), but I need to figure out how
      to add a point to draw lines from it to its nearest neighbors.
'''

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go

from knn import *

matplotlib.use("TkAgg")

num_classes = 0
num_neighbors = 0
num_points = 0
metric = 0


# Buttons
layout = [
    [sg.Text("K-Nearest Neighbors Visualizer")],
    [sg.Canvas(key="-CANVAS-")],
    [sg.Text('Metric', justification='left'), sg.Button("L1"), sg.Button("L2")],
    [sg.Text('Num classes', justification='left'), sg.Button("2", key='num_classes2'), sg.Button("3", key='num_classes3'), sg.Button("4", key='num_classes4'), sg.Button("5", key='num_classes5')],
    [sg.Text('Num Neighbors (K)', justification='left'), sg.Button("2", key='num_neighbors2'), sg.Button("3", key='num_neighbors3'), sg.Button("4", key='num_neighbors4'), sg.Button("5", key='num_neighbors5'), sg.Button("6", key='num_neighbors6'), sg.Button("7", key='num_neighbors7')],
    [sg.Text('Num Points', justification='left'), sg.Button("20", key='num_points20'), sg.Button("30", key='num_points30'), sg.Button("40", key='num_points40'), sg.Button("50", key='num_points50'), sg.Button("60", key='num_points60')],
    [sg.Text('Apply', justification='left'), sg.Button("Apply", key='apply')]
]


# Create the window
window = sg.Window(
    "KNN Visualizer",
    layout,
    location=(0, 0),
    finalize=True,
    element_justification="center",
    font="Helvetica 18",
)
          

while True:          
    event, values = window.Read()
    if event in (None, 'Exit'):
        break
    if event == 'apply':
        main(num_classes, num_neighbors, num_points, metric)
    if event == 'L1':
        metric = 1
    elif event == 'L2':
        metric = 2
    elif event == 'num_classes2':
        num_classes = 2
    elif event == 'num_classes3':
        num_classes = 3
    elif event == 'num_classes4':
        num_classes = 4
    elif event == 'num_classes5':
        num_classes = 5
    elif event == 'num_neighbors2':
        num_neighbors = 2
    elif event == 'num_neighbors3':
        num_neighbors = 3
    elif event == 'num_neighbors4':
        num_neighbors = 4
    elif event == 'num_neighbors5':
        num_neighbors = 5
    elif event == 'num_neighbors6':
        num_neighbors = 6
    elif event == 'num_neighbors7':
        num_neighbors = 7
    elif event == 'num_points20':
        num_points = 20
    elif event == 'num_points30':
        num_points = 30
    elif event == 'num_points40':
        num_points = 40
    elif event == 'num_points50':
        num_points = 50 
    elif event == 'num_points60':
        num_points = 60


event, values = window.read()

window.close()
