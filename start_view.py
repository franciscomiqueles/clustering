from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter as Tk
from scipy.stats import norm
from Clust import Clust
from clust_view import clust_view
from cell_view import cell_view

class start_view():
    
    def __init__(self, master, clust):
        self.clust = clust
        self.master = master
        self.frame = Tk.LabelFrame(master, text = "Select EXP", height= 50, width= 50, bd = 5)
        self.frame.grid(row = 1, column = 1, columnspan= 10, padx= 10, pady =10)


        self.load_data_button = None
        self.cell_button = None
        self.clust_button = None

        self.i = 0
        self.j = 0
        self.variables = []
        

    def load_data(self, value):
        new = []
        for j in range(len(value)):
            if(value[j].get()):
                new.append(self.clust.clust_exp[j][1])
        if bool(new):
            print(new)
            self.clust.get_data(new)
            self.clust.compute_spike_distance()
            self.cell_button = Tk.Button(self.master, text = "Cell View", command= self.start_cell_view)
            self.cell_button.grid(row = 2, column = 1, padx = 10)
            self.clust_button = Tk.Button(self.master, text = "Clust View", command= self.start_clust_view)
            self.clust_button.grid(row = 2 , column = 2)
        else:
            print("Error, try again")
        
    def start_clust_view(self):
        root = Tk.Tk()
        app = clust_view(root, self.clust)
        app.start_gui()
        root.title( "Clust View")
        root.update()
        root.deiconify()
        root.mainloop()
    
    def start_cell_view(self):
        root = Tk.Tk()
        app = cell_view(root, self.clust)
        app.start_gui()
        root.title( "Clust View")
        root.update()
        root.deiconify()
        root.mainloop()


    def start_gui(self):
        self.clust.read_params()
        self.clust.set_folder()
        for exp in range(len(self.clust.clust_exp)):
            r = Tk.IntVar()
            self.variables.append(r)
            self.exp1 = Tk.Checkbutton(self.frame, text = "{}".format(self.clust.clust_exp[exp][0]), variable= r)
            self.exp1.grid(row = self.i, column = self.j)
            if(self.i == 2):
                self.i = 0
                self.j = self.j + 1
            else:
                self.i = self.i + 1
            
        self.load_data_button = Tk.Button(self.frame, text = "Load Data", command=lambda: self.load_data(self.variables))
        self.load_data_button.grid(row = 1, column = self.j + 1)
        self.cell_button = Tk.Button(self.master, text = "Cell View", state = Tk.DISABLED)
        self.cell_button.grid(row = 2, column = 1, padx = 10)
        self.clust_button = Tk.Button(self.master, text = "Clust View", state = Tk.DISABLED)
        self.clust_button.grid(row = 2 , column = 2)
        

if __name__ == '__main__':
    clust = Clust("clust")
    root = Tk.Tk()
    app = start_view(root, clust)
    app.start_gui()
    root.title( "GUI")
    root.update()
    root.deiconify()
    root.mainloop()