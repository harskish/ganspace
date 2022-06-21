import matplotlib.pyplot as plt
import pickle
import os
import sys

def show_figure(fig):
    # create a dummy figure and use its
    # manager to display "fig"
    dummy = plt.figure(num=1)
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    plt.axis('equal')
    plt.show()

plt.switch_backend('TkAgg')

path = sys.argv[1]
if os.path.isfile(path):
    if(path.split('.')[-1] == 'pickle'):
        print("Loading", path.split('/')[-1])
        figx = pickle.load(open(path, 'rb'))
        show_figure(figx)
