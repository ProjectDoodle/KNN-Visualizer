import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go

matplotlib.use("TkAgg")

def func(message):
    print(mess

fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
t = np.arange(0, 3, .01)
fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))


# Graph
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

# Buttons
layout = [
    [sg.Text("K-Nearest Neighbors Visualizer")],
    [sg.Canvas(key="-CANVAS-")],
    [sg.Text('Metric', justification='left'), sg.Button("L1"), sg.Button("L2")],
    [sg.Text('Num classes', justification='left'), sg.Button("2", key='num_classes2'), sg.Button("3", key='num_classes3'), sg.Button("4", key='num_classes4'), sg.Button("5", key='num_classes5')],
    [sg.Text('Num Neighbors (K)', justification='left'), sg.Button("2", key='num_neighbors2'), sg.Button("3", key='num_neighbors3'), sg.Button("4", key='num_neighbors4'), sg.Button("5", key='num_neighbors5'), sg.Button("6", key='num_neighbors6'), sg.Button("7", key='num_neighbors7')],
    [sg.Text('Num Points', justification='left'), sg.Button("20", key='num_points20'), sg.Button("30", key='num_points30'), sg.Button("40", key='num_points40'), sg.Button("50", key='num_points50'), sg.Button("60", key='num_points60')],
    [sg.Text('Passes', size=(8, 1)), sg.Spin(values=[i for i in range(1, 1000)], initial_value=20, size=(6, 1)),
                           sg.Text('Steps', size=(8, 1), pad=((7, 3))), sg.Spin(values=[i for i in range(1, 1000)], initial_value=20, size=(6, 1))],
                          [sg.Text('ooa', size=(8, 1)), sg.Input(default_text='6', size=(8, 1)), sg.Text('nn', size=(8, 1)),
                           sg.Input(default_text='10', size=(10, 1))],
                          [sg.Text('q', size=(8, 1)), sg.Input(default_text='ff', size=(8, 1)), sg.Text('ngram', size=(8, 1)),
                           sg.Input(default_text='5', size=(10, 1))],
                          [sg.Text('l', size=(8, 1)), sg.Input(default_text='0.4', size=(8, 1)), sg.Text('Layers', size=(8, 1)),
                           sg.Drop(values=('BatchNorm', 'other'))], 
]


# Create the form and show it without the plot
window = sg.Window(
    "KNN Visualizer",
    layout,
    location=(0, 0),
    finalize=True,
    element_justification="center",
    font="Helvetica 18",
)
          
    # Event Loop
  while True:             # Event Loop
    event, values = window.Read()
    if event in (None, 'Exit'):
        break
    if event == 'L1':
        func('Pressed button L1')
    elif event == 'L2':
        func('Pressed button L2')    
    elif event == 'num_classes2':
        func('Pressed button 2')
    elif event == 'num_classes3':
        func('Pressed button 3')
    elif event == 'num_classes4':
        func('Pressed button 4')
    elif event == 'num_classes5':
        func('Pressed button 5') 
    elif event == 'num_neighbors2':
        func('Pressed button 2')
    elif event == 'num_neighbors3':
        func('Pressed button 3')  
    elif event == 'num_neighbors4':
        func('Pressed button 4')
    elif event == 'num_neighbors5':
        func('Pressed button 5')
    elif event == 'num_neighbors6':
        func('Pressed button 6')
    elif event == 'num_neighbors7':
        func('Pressed button 7')
    elif event == 'num_points20':
        func('Pressed button 20')
    elif event == 'num_points30':
        func('Pressed button 30') 
    elif event == 'num_points40':
        func('Pressed button 40')   
    elif event == 'num_points50':
        func('Pressed button 50')  
    elif event == 'num_points60':
        func('Pressed button 60')

# Add the plot to the window
draw_figure(window["-CANVAS-"].TKCanvas, fig)

event, values = window.read()

window.close()
