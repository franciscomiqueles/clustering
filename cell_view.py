from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter as Tk
from Clust import Clust

class cell_view():
    
    def __init__(self, master, clust):
        self.clust = clust
        self.master = master
        self.frame = Tk.Frame(master)

        self.fig, self.axes = plt.subplots(1, 2, dpi = 80, figsize=(10,2))
        self.fig_psth, self.axes_psth = plt.subplots(1, 2, dpi = 80, figsize=(10,2))

        self.frame_trials = Tk.Frame( master )
        self.frame_trials.grid(row = 3 , column = 1, columnspan = 3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_trials)
        self.canvas.get_tk_widget().grid(row = 3, column = 1, columnspan = 3)

        self.frame_psth = Tk.Frame( master )
        self.frame_psth.grid(row = 5 , column = 1, columnspan = 3)

        self.canvas_psth = FigureCanvasTkAgg(self.fig_psth, master=self.frame_psth)
        self.canvas_psth.get_tk_widget().grid(row = 5, column = 1, columnspan = 3)

        self.exp_label = None
        self.unit_label = None

        self.next_button_trials = None
        self.back_button_trials = None

        self.next_button_psth = None
        self.back_button_psth = None

        self.trial_status = None
        self.psth_status = None
    
    def set_ax_gui(self, i, j, psth, trial):

        if(i == 0):
            self.back_button_psth = Tk.Button(self.master, text ="<<", state = Tk.DISABLED, command = lambda : self.set_ax_gui(i-1, j, 1, 1))
            self.back_button_psth.grid(row = 4, column=1)
        else:
            self.back_button_psth = Tk.Button(self.master, text ="<<", command = lambda : self.set_ax_gui(i-1, j, 1, 1))
            self.back_button_psth.grid(row = 4, column=1)
        if(j == 0):
            self.back_button_trials = Tk.Button(self.master, text ="<<", state = Tk.DISABLED, command = lambda: self.set_ax_gui(i, j-1, 0, 1))
            self.back_button_trials.grid(row = 2, column=1)
        else:
            self.back_button_trials = Tk.Button(self.master, text ="<<", command = lambda: self.set_ax_gui(i, j-1, 0, 1))
            self.back_button_trials.grid(row = 2, column=1)
        if(i == len(self.clust.cells)):
            self.next_button_psth = Tk.Button(self.master, text =">>", state = Tk.DISABLED, command = lambda : self.set_ax_gui(i+1, j, 1, 1))
            self.next_button_psth.grid(row = 4, column=3)
        else:
            self.next_button_psth = Tk.Button(self.master, text =">>", command = lambda : self.set_ax_gui(i+1, j, 1, 1))
            self.next_button_psth.grid(row = 4, column=3)
        if(j == 20):
            self.next_button_trials = Tk.Button(self.master, text =">>", state = Tk.DISABLED, command = lambda: self.set_ax_gui(i, j+1, 0, 1))
            self.next_button_trials.grid(row = 2, column=3)
        else:
            self.next_button_trials = Tk.Button(self.master, text =">>", command = lambda: self.set_ax_gui(i, j+1, 0 , 1))
            self.next_button_trials.grid(row = 2, column=3)

        self.psth_status = Tk.Label(self.master, text = "Cell {} of {}".format(i + 1, len(self.clust.cells)))
        self.trial_status = Tk.Label(self.master, text = "Trial {} of {}".format(j + 1, 21)) 
        self.psth_status.grid(row = 4, column = 2)
        self.trial_status.grid(row = 2, column = 2)

        if(psth):
            self.exp_label = Tk.Label(self.master, text = "{}".format(self.clust.cells[i].exp_unit[0]))
            self.unit_label = Tk.Label(self.master, text = "{}".format(self.clust.cells[i].exp_unit[1]))
            self.exp_label.grid(row = 1, column = 1)
            self.unit_label.grid(row = 1, column =2, columnspan = 3)
            self.axes_psth[0].clear()
            self.axes_psth[1].clear()
            self.axes_psth[0].plot(self.clust.color_time, self.clust.cells[i].color_psth)
            self.axes_psth[0].set_title("Color Response")
            self.axes_psth[1].plot(self.clust.chirp_time, self.clust.cells[i].chirp_psth)
            self.axes_psth[1].set_title("Chirp Response")
            self.canvas_psth = FigureCanvasTkAgg(self.fig_psth, master=self.frame_psth)
            self.canvas_psth.get_tk_widget().grid(row=5, column=1, columnspan = 3)
            

        if(trial):
            self.axes[0].clear()
            self.axes[1].clear()
            self.axes[0].plot(self.clust.color_time, self.clust.cells[i].flash_trials_psth[j])
            self.axes[0].set_title("Color Response")
            self.axes[1].plot(self.clust.chirp_time, self.clust.cells[i].chirp_trials_psth[j])
            self.axes[1].set_title("Chirp Response")
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_trials)
            self.canvas.get_tk_widget().grid(row=3, column=1, columnspan = 3)
    
    def start_gui(self):

        self.axes[0].plot(self.clust.color_time, self.clust.cells[0].flash_trials_psth[0])
        self.axes[0].set_title("Color Response")
        self.axes[1].plot(self.clust.chirp_time, self.clust.cells[0].chirp_trials_psth[0])
        self.axes[1].set_title("Chirp Response")

        self.axes_psth[0].plot(self.clust.color_time, self.clust.cells[0].color_psth)
        self.axes_psth[0].set_title("Color Response")
        self.axes_psth[1].plot(self.clust.chirp_time, self.clust.cells[0].chirp_psth)
        self.axes_psth[1].set_title("Chirp Response")
        
        self.exp_label = Tk.Label(self.master, text = "{}".format(self.clust.cells[0].exp_unit[0]))
        self.unit_label = Tk.Label(self.master, text = "{}".format(self.clust.cells[0].exp_unit[1]))
        self.exp_label.grid(row = 1, column = 1)
        self.unit_label.grid(row = 1, column =2, columnspan = 3)

        self.next_button_psth = Tk.Button(self.master, text =">>", command = lambda : self.set_ax_gui(1, 0, 1, 1))
        self.next_button_psth.grid(row = 4, column=3)

        self.next_button_trials = Tk.Button(self.master, text =">>", command = lambda: self.set_ax_gui(1, 1, 0, 1))
        self.next_button_trials.grid(row = 2, column=3)

        self.back_button_psth = Tk.Button(self.master, text ="<<", state = Tk.DISABLED, command = lambda : self.set_ax_gui(0, 0, 1, 1))
        self.back_button_psth.grid(row = 4, column=1)

        self.back_button_trials = Tk.Button(self.master, text ="<<", state = Tk.DISABLED, command = lambda: self.set_ax_gui(0, 0, 0, 1))
        self.back_button_trials.grid(row = 2, column=1)

        self.psth_status = Tk.Label(self.master, text = "Cell {} of {}".format(1, len(self.clust.cells)))
        self.trial_status = Tk.Label(self.master, text = "Trial {} of {}".format(1, 21))
        self.psth_status.grid(row = 4, column = 2)
        self.trial_status.grid(row = 2, column = 2)

if __name__ == '__main__':
    clust = Clust("clust")
    root = Tk.Tk()
    app = cell_view(root, clust)
    app.start_gui()
    root.title( "Gráficos")
    root.update()
    root.deiconify()
    root.mainloop()