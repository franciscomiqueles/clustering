from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter as Tk
from scipy.stats import norm
from Clust import Clust

class clust_view():
    
    def __init__(self, master, clust):
        self.clust = clust
        self.master = master
        self.frame = Tk.Frame(master)
        self.fig, self.axes = plt.subplots(1, 3, dpi = 115, figsize=(10,2))

        self.frame_trials = Tk.LabelFrame( master , text = "Clusters")
        self.frame_trials.grid(row = 4 , column = 1, columnspan = 4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_trials)
        self.canvas.get_tk_widget().grid(row = 4, column = 1, columnspan = 4)
        
    
        self.frame_den = Tk.LabelFrame( master , text = "Dendogram")
        self.frame_den.grid(row = 1 , column = 1, columnspan = 3, rowspan=2)
        self.frame_tsne = Tk.LabelFrame( master , text = "TSNE")
        self.frame_tsne.grid(row = 1 , column = 4, pady= 10, padx=20)
        self.frame_mut = Tk.LabelFrame( master , text = "Mutual Info")
        self.frame_mut.grid(row = 2 , column = 4, pady=10, padx=20)

        self.canvas_den = None
        self.canvas_tsne = None
        self.canvas_mut = None
        self.trial_status = None

        self.clust_flash_mean = None
        self.clust_chirp_mean = None
        self.clust_bias_fix = None
        
        self.dend = None
        self.mut = None
        self.tsne = None

    def set_ax_gui(self, i):

        if(i == 0):
            self.back_button_trials = Tk.Button(self.master, text ="<<", state = Tk.DISABLED, command = lambda: self.set_ax_gui(i-1))
            self.back_button_trials.grid(row = 3, column=1)
        else:
            self.back_button_trials = Tk.Button(self.master, text ="<<", command = lambda: self.set_ax_gui(i-1))
            self.back_button_trials.grid(row = 3, column=1)
        if(i == self.clust.n_cluster - 1):
            self.next_button_trials = Tk.Button(self.master, text =">>", state = Tk.DISABLED, command = lambda: self.set_ax_gui(i+1))
            self.next_button_trials.grid(row = 3, column=3)
        else:
            self.next_button_trials = Tk.Button(self.master, text =">>", command = lambda: self.set_ax_gui(i+1))
            self.next_button_trials.grid(row = 3, column=3)

        
        self.trial_status = Tk.Label(self.master, text = "Cluster {} of {}".format(i + 1, self.clust.n_cluster)) 
        self.trial_status.grid(row = 3, column = 2)

        self.axes[0].clear()
        self.axes[1].clear()
        self.axes[2].clear()

        self.axes[0].plot(self.clust.flash_time, self.clust_flash_mean[i])
        self.axes[0].set_title("Flash Response")
        self.axes[1].plot(self.clust.chirp_time, self.clust_chirp_mean[i])
        self.axes[1].set_title("Chirp Response")
        self.axes[2].hist(self.clust_bias_fix[i], bins = 25, range=[-1, 1])
        self.axes[2].set_title("BIAS Histogram")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_trials)
        self.canvas.get_tk_widget().grid(row = 4, column = 1, columnspan = 3)
        mean,std = norm.fit(self.clust_bias_fix[i])
        if std==0:
             std=0.25
            
        x = np.linspace(-1, 1, 100)
        y = norm.pdf(x, mean, std)

        self.axes[2].plot(x, y)
        
    def show_figure(self):
        root = Tk.Tk()
        fig = self.clust.get_bias_histogram()
        frame = Tk.Frame(root)
        frame.grid(row=1, column=1)
        canvas_fig = FigureCanvasTkAgg(fig, master=frame)
        canvas_fig.get_tk_widget().grid(row = 1, column = 1)
        root.title( "Figure")
        root.update()
        root.deiconify()
        root.mainloop()
        
    def save_data(self):
        self.clust.save_data()

    def start_gui(self):
        self.dend , self.mut = self.clust.get_dendogram()
        self.tsne = self.clust.get_tsne()
        self.clust_flash_mean, self.clust_chirp_mean, self.clust_bias_fix = self.clust.get_clust_mean()

        self.axes[0].plot(self.clust.flash_time, self.clust_flash_mean[0])
        self.axes[0].set_title("Flash Response")
        self.axes[1].plot(self.clust.chirp_time, self.clust_chirp_mean[0])
        self.axes[1].set_title("Chirp Response")
        self.axes[2].hist(self.clust_bias_fix[0], bins = 25, range=[-1, 1])
        self.axes[2].set_title("BIAS Histogram")
        mean,std = norm.fit(self.clust_bias_fix[0])
        if std==0:
             std=0.25
            
        x = np.linspace(-1, 1, 100)
        y = norm.pdf(x, mean, std)
        self.axes[2].plot(x, y)
        
        self.next_button_trials = Tk.Button(self.master, text =">>", command = lambda: self.set_ax_gui(1))
        self.next_button_trials.grid(row = 3, column=3)

        self.back_button_trials = Tk.Button(self.master, text ="<<", state = Tk.DISABLED, command = lambda: self.set_ax_gui(1))
        self.back_button_trials.grid(row = 3, column=1)

        self.trial_status = Tk.Label(self.master, text = "Cluster {} of {}".format(1, self.clust.n_cluster))
        self.trial_status.grid(row = 3, column = 2)
        
        self.canvas_den = FigureCanvasTkAgg(self.dend, master=self.frame_den)
        self.canvas_den.get_tk_widget().grid(row = 1, column = 1, columnspan=3, rowspan=2)
        self.canvas_tsne = FigureCanvasTkAgg(self.tsne, master=self.frame_tsne)
        self.canvas_tsne.get_tk_widget().grid(row = 1, column = 4)
        self.canvas_mut = FigureCanvasTkAgg(self.mut, master=self.frame_mut)
        self.canvas_mut.get_tk_widget().grid(row = 2, column = 4)
        self.figure_button = Tk.Button(self.master, text ="Show Figure", command= self.show_figure)
        self.figure_button.grid(row = 3, column=4)
        self.save_button = Tk.Button(self.master, text ="Save Figure", command= self.save_data)
        self.save_button.grid(row = 3, column=5)

        
    

if __name__ == '__main__':
    clust = Clust("clust")
    root = Tk.Tk()
    app = clust_view(root, clust)
    app.start_gui()
    root.title( "Gráficos")
    root.update()
    root.deiconify()
    root.mainloop()