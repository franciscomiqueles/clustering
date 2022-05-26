from __future__ import absolute_import, print_function

from configparser import ConfigParser, ExtendedInterpolation
from turtle import color
                                                                                                                                                                                
import scipy.io
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statistics
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
import pyspike
import os
import joblib
import warnings
import shutil
import multiprocessing as mp

from ipywidgets import IntSlider, interact, Dropdown, fixed
from spikelib import spiketools as spkt

import scipy.cluster as cluster
import scipy.spatial.distance as distance
from scipy.spatial.distance import squareform
import sklearn.metrics.cluster as metrics
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from scipy.interpolate import interp1d
from scipy.stats import norm
from tqdm import tqdm
from chirp import get_chirp_subevents, get_pop_response, chirp_def_args, chirp_generator
from functions import format_to_pyspike, get_cell_trials, compute_ISI_on_flat_pair, compute_SPIKE_on_flat_pair, flat_unit_pairs, format_to_pyspike, time_to_amp, get_fit
from Cell import Cell
import json
from extract_features import plot_features, plot_resp
config = ConfigParser(interpolation=ExtendedInterpolation())


class Clust:

    def __init__(self, folder):

        self.exp_unit = []
        self.spiketimes = []
        self.chirp_psth = []
        self.flash_psth = []
        self.bias_index = []
        self.cells = []
        
        self.pcells = None
        self.params = None

        self.cls_folder = folder

        self.chirp_time = None
        self.flash_time = None
        self.color_time = None

        self.spike_dst = None
        self.isi_dst = None

        self.n_cluster = None
        self.cls_spike = None 

        self.clust_exp = []
        self.n = None

    def read_params(self):
        with open("params.json") as p:
            self.params = json.load(p)
    
        # Create output folder if it does not exist
        if os.path.isdir(self.params['Output']) == False:
            os.mkdir(self.params['Output'])
        
        for exp_path in self.params['Experiments']:
            os.chdir(exp_path)
            cfg_file = 'config.ini'
            config.read(cfg_file)
            config.set('PROJECT', 'path', '{}/'.format(os.getcwd()))
            with open(cfg_file, 'w+') as configfile:
                config.write(configfile)
            config.read(cfg_file)
            exp = config['EXP']['name']
            self.clust_exp.append([exp, exp_path])

    def ensembles_analysis(self, exp_path):
        file = os.path.join(exp_path[1], '{}_ensembles.csv'.format(exp_path[0])) 
        isdir = os.path.isfile(file)
        it = 0
        if(isdir):
            df_feat = pd.read_csv(file, index_col=0)
            self.n = len(df_feat.columns)-9
            for cell in self.cells:
                it = cell.get_ensembles(file, self.n, it)
        for en in range(1, self.n+1):
            self.get_exp_analysis(en)

    def create_csv(self):
        csv_def = []
        names = []
        for en in range(self.n):
            names.append("Ensemble_{}".format(en+1))
            csv = []
            asam = []
            for cell in self.cells:
                if(cell.ensembles_on and cell.ensembles_on[en]):
                    if(cell.type == 0):
                        arg = [cell.exp_unit[1], "ON", cell.on_info[0], np.nan, cell.on_info[1], np.nan, cell.on_info[2], np.nan, cell.max_freq_response_on, np.nan, cell.bias_index]
                        asam.append(arg)
            if(asam):    
                asam = np.asarray(asam)
                df = pd.DataFrame(asam, columns = ['unit','flash_type','freq_cut_on [Hz]', 'freq_cut_off [Hz]', 'sust_index_on [-]', 'sust_index_off [-]', 'delay_on [s]', 'delay_off [s]', 'max_response_on [Hz]', 'max_response_off [Hz]', 'bias_index [-]'])
                for col in df.columns:
                    if(col != "unit" and col != "flash_type"):
                        df[col] = pd.to_numeric(df[col], downcast="float", errors='coerce')
                arg1 = ['', 'Mean', np.round(df['freq_cut_on [Hz]'].mean(), decimals =14 ), np.round(df['freq_cut_off [Hz]'].mean(), decimals =14 ), np.round(df['sust_index_on [-]'].mean(), decimals =14 ), np.round(df['sust_index_off [-]'].mean(), decimals =14 ), np.round(df['delay_on [s]'].mean(), decimals =14 ), np.round(df['delay_off [s]'].mean(), decimals =14 ), np.round(df['max_response_on [Hz]'].mean(), decimals =14 ), np.round(df['max_response_off [Hz]'].mean(), decimals =14 ), np.round(df['bias_index [-]'].mean(), decimals =4 )]
                arg1 = np.asarray(arg1)
                df1 = pd.DataFrame([arg1], columns = ['unit','flash_type','freq_cut_on [Hz]', 'freq_cut_off [Hz]', 'sust_index_on [-]', 'sust_index_off [-]', 'delay_on [s]', 'delay_off [s]', 'max_response_on [Hz]', 'max_response_off [Hz]', 'bias_index [-]'])
                arg2 = ['', 'Std', np.round(df['freq_cut_on [Hz]'].std(), decimals =14 ), np.round(df['freq_cut_off [Hz]'].std(), decimals =14 ), np.round(df['sust_index_on [-]'].std(), decimals =14 ), np.round(df['sust_index_off [-]'].std(), decimals =14 ), np.round(df['delay_on [s]'].std(), decimals =14 ), np.round(df['delay_off [s]'].std(), decimals =14 ), np.round(df['max_response_on [Hz]'].std(), decimals =14 ), np.round(df['max_response_off [Hz]'].std(), decimals =14 ), np.round(df['bias_index [-]'].std(), decimals =4 )]
                arg2 = np.asarray(arg2)
                df2 = pd.DataFrame([arg2], columns = ['unit','flash_type','freq_cut_on [Hz]', 'freq_cut_off [Hz]', 'sust_index_on [-]', 'sust_index_off [-]', 'delay_on [s]', 'delay_off [s]', 'max_response_on [Hz]', 'max_response_off [Hz]', 'bias_index [-]'])
                arg3 = ['', 'Count', df['unit'].count(), '', '', '', '', '', '', '', '']
                arg3 = np.asarray(arg3)
                df3 = pd.DataFrame([arg3], columns = ['unit','flash_type','freq_cut_on [Hz]', 'freq_cut_off [Hz]', 'sust_index_on [-]', 'sust_index_off [-]', 'delay_on [s]', 'delay_off [s]', 'max_response_on [Hz]', 'max_response_off [Hz]', 'bias_index [-]'])
                df = pd.concat([df, df3, df1, df2], axis = 0, ignore_index=True)
                csv.append(df)
            asam = []
            for cell in self.cells:
                if(cell.ensembles_off and cell.ensembles_off[en]):
                    if(cell.type == 1):
                        arg = [cell.exp_unit[1], "OFF", np.nan, cell.off_info[0], np.nan, cell.off_info[1], np.nan, cell.off_info[2], np.nan, cell.max_freq_response_off, cell.bias_index]
                        asam.append(arg)
            if(asam):   
                asam = np.asarray(asam)
                df = pd.DataFrame(asam, columns = ['unit','flash_type','freq_cut_on [Hz]', 'freq_cut_off [Hz]', 'sust_index_on [-]', 'sust_index_off [-]', 'delay_on [s]', 'delay_off [s]', 'max_response_on [Hz]', 'max_response_off [Hz]', 'bias_index [-]'])
                for col in df.columns:
                    if(col != "unit" and col != "flash_type"):
                        df[col] = pd.to_numeric(df[col], downcast="float", errors='coerce')
                arg1 = ['', 'Mean', np.round(df['freq_cut_on [Hz]'].mean(), decimals =14 ), np.round(df['freq_cut_off [Hz]'].mean(), decimals =14 ), np.round(df['sust_index_on [-]'].mean(), decimals =14 ), np.round(df['sust_index_off [-]'].mean(), decimals =14 ), np.round(df['delay_on [s]'].mean(), decimals =14 ), np.round(df['delay_off [s]'].mean(), decimals =14 ), np.round(df['max_response_on [Hz]'].mean(), decimals =14 ), np.round(df['max_response_off [Hz]'].mean(), decimals =14 ), np.round(df['bias_index [-]'].mean(), decimals =4 )]
                arg1 = np.asarray(arg1)
                df1 = pd.DataFrame([arg1], columns = ['unit','flash_type','freq_cut_on [Hz]', 'freq_cut_off [Hz]', 'sust_index_on [-]', 'sust_index_off [-]', 'delay_on [s]', 'delay_off [s]', 'max_response_on [Hz]', 'max_response_off [Hz]', 'bias_index [-]'])
                arg2 = ['', 'Std', np.round(df['freq_cut_on [Hz]'].std(), decimals =14 ), np.round(df['freq_cut_off [Hz]'].std(), decimals =14 ), np.round(df['sust_index_on [-]'].std(), decimals =14 ), np.round(df['sust_index_off [-]'].std(), decimals =14 ), np.round(df['delay_on [s]'].std(), decimals =14 ), np.round(df['delay_off [s]'].std(), decimals =14 ), np.round(df['max_response_on [Hz]'].std(), decimals =14 ), np.round(df['max_response_off [Hz]'].std(), decimals =14 ), np.round(df['bias_index [-]'].std(), decimals =4 )]
                arg2 = np.asarray(arg2)
                df2 = pd.DataFrame([arg2], columns = ['unit','flash_type','freq_cut_on [Hz]', 'freq_cut_off [Hz]', 'sust_index_on [-]', 'sust_index_off [-]', 'delay_on [s]', 'delay_off [s]', 'max_response_on [Hz]', 'max_response_off [Hz]', 'bias_index [-]'])
                arg3 = ['', 'Count', df['unit'].count(), '', '', '', '', '', '', '', '']
                arg3 = np.asarray(arg3)
                df3 = pd.DataFrame([arg3], columns = ['unit','flash_type','freq_cut_on [Hz]', 'freq_cut_off [Hz]', 'sust_index_on [-]', 'sust_index_off [-]', 'delay_on [s]', 'delay_off [s]', 'max_response_on [Hz]', 'max_response_off [Hz]', 'bias_index [-]'])
                df = pd.concat([df, df3, df1, df2], axis = 0, ignore_index=True)
                csv.append(df)
            asam = []
            for cell in self.cells:
                if((cell.ensembles_off and cell.ensembles_off[en]) and (cell.ensembles_on and cell.ensembles_on[en])):
                    if(cell.type == 2):
                        arg = [cell.exp_unit[1], "ON/OFF", cell.on_info[0], cell.off_info[0], cell.on_info[1], cell.off_info[1], cell.on_info[2], cell.off_info[2], cell.max_freq_response_on, cell.max_freq_response_off, cell.bias_index]
                        asam.append(arg)
            if(asam):   
                asam = np.asarray(asam)
                df = pd.DataFrame(asam, columns = ['unit','flash_type','freq_cut_on [Hz]', 'freq_cut_off [Hz]', 'sust_index_on [-]', 'sust_index_off [-]', 'delay_on [s]', 'delay_off [s]', 'max_response_on [Hz]', 'max_response_off [Hz]', 'bias_index [-]'])
                for col in df.columns:
                    if(col != "unit" and col != "flash_type"):
                        df[col] = pd.to_numeric(df[col], downcast="float", errors='coerce')
                arg1 = ['', 'Mean', np.round(df['freq_cut_on [Hz]'].mean(), decimals =14 ), np.round(df['freq_cut_off [Hz]'].mean(), decimals =14 ), np.round(df['sust_index_on [-]'].mean(), decimals =14 ), np.round(df['sust_index_off [-]'].mean(), decimals =14 ), np.round(df['delay_on [s]'].mean(), decimals =14 ), np.round(df['delay_off [s]'].mean(), decimals =14 ), np.round(df['max_response_on [Hz]'].mean(), decimals =14 ), np.round(df['max_response_off [Hz]'].mean(), decimals =14 ), np.round(df['bias_index [-]'].mean(), decimals =4 )]
                arg1 = np.asarray(arg1)
                df1 = pd.DataFrame([arg1], columns = ['unit','flash_type','freq_cut_on [Hz]', 'freq_cut_off [Hz]', 'sust_index_on [-]', 'sust_index_off [-]', 'delay_on [s]', 'delay_off [s]', 'max_response_on [Hz]', 'max_response_off [Hz]', 'bias_index [-]'])
                arg2 = ['', 'Std', np.round(df['freq_cut_on [Hz]'].std(), decimals =14 ), np.round(df['freq_cut_off [Hz]'].std(), decimals =14 ), np.round(df['sust_index_on [-]'].std(), decimals =14 ), np.round(df['sust_index_off [-]'].std(), decimals =14 ), np.round(df['delay_on [s]'].std(), decimals =14 ), np.round(df['delay_off [s]'].std(), decimals =14 ), np.round(df['max_response_on [Hz]'].std(), decimals =14 ), np.round(df['max_response_off [Hz]'].std(), decimals =14 ), np.round(df['bias_index [-]'].std(), decimals =4 )]
                arg2 = np.asarray(arg2)
                df2 = pd.DataFrame([arg2], columns = ['unit','flash_type','freq_cut_on [Hz]', 'freq_cut_off [Hz]', 'sust_index_on [-]', 'sust_index_off [-]', 'delay_on [s]', 'delay_off [s]', 'max_response_on [Hz]', 'max_response_off [Hz]', 'bias_index [-]'])
                arg3 = ['', 'Count', df['unit'].count(), '', '', '', '', '', '', '', '']
                arg3 = np.asarray(arg3)
                df3 = pd.DataFrame([arg3], columns = ['unit','flash_type','freq_cut_on [Hz]', 'freq_cut_off [Hz]', 'sust_index_on [-]', 'sust_index_off [-]', 'delay_on [s]', 'delay_off [s]', 'max_response_on [Hz]', 'max_response_off [Hz]', 'bias_index [-]'])
                df = pd.concat([df, df3, df1, df2], axis = 0, ignore_index=True)
                csv.append(df)
            csv = pd.concat(csv)
            csv_def.append(csv)
        writer = pd.ExcelWriter('{}/ensembles_analysis.xlsx'.format(self.cls_folder), engine='xlsxwriter')
        for x, n in zip(csv_def , names):
            x.to_excel(writer, n)
            for column in x:
                column_length = max(x[column].astype(str).map(len).max(), len(column))
                col_idx = x.columns.get_loc(column)
                writer.sheets[n].set_column(col_idx, col_idx, column_length)
        writer.save()
       
    def get_clust_mean(self):
        clust_flash_mean = []
        clust_chirp_mean = []
        clust_bias_fix = []
        self.bias_index = np.asarray(self.bias_index)
        for i in range(1,self.n_cluster +1):
            cls_units = np.asarray(self.exp_unit, dtype='<U9')[self.cls_spike == i]
            cls_chirp = self.chirp_psth[self.cls_spike == i]
            cls_flash = self.flash_psth[self.cls_spike == i]
            cls_bias = self.bias_index[self.cls_spike == i]
            flash_mean = 0
            chirp_mean = 0
            bias_mean = []
            for (uid, chirp, flash, bias) in zip(cls_units, cls_chirp, cls_flash, cls_bias):
                flash_mean += flash
                chirp_mean += chirp
                if (not np.isnan(bias)):
                    bias_mean.append(bias)
            flash_mean=flash_mean/len(cls_units)
            chirp_mean=chirp_mean/len(cls_units)
            clust_flash_mean.append(flash_mean)
            clust_chirp_mean.append(chirp_mean)
            clust_bias_fix.append(bias_mean)
        return clust_flash_mean, clust_chirp_mean, clust_bias_fix 
    
    def get_exp_analysis(self, number):
        plt.figure(figsize=(16,5))
        plot_stims = [0,1]
        plot_widths = [1, 0.5, 1.3, 1.3, 1.3, 1.3, 1.3]
        bias = []
        gs = gridspec.GridSpec(2, 7, width_ratios = plot_widths, wspace=0.45, hspace=0.5)
        cut_on = []
        sust_on = []
        delay_on = []
        cut_off = []
        sust_off = []
        delay_off = []
        max_on = []
        max_off = []
        for cell in self.cells:
            if(cell.ensembles_on and cell.ensembles_on[number-1] == 1):  
                if(cell.on_info):
                    if (not np.isnan(cell.on_info[0])):
                        cut_on.append(int(cell.on_info[0]))
                    if (not np.isnan(cell.on_info[1])):
                        sust_on.append(cell.on_info[1])
                    if (not np.isnan(cell.on_info[2])):
                        delay_on.append(cell.on_info[2])
                if(cell.max_freq_response_on is not None):
                    if (not np.isnan(cell.max_freq_response_on)):
                        max_on.append(int(cell.max_freq_response_on))
            if(cell.ensembles_off and cell.ensembles_off[number-1]==1):  
                if(cell.off_info):
                    if (not np.isnan(cell.off_info[0])):
                        cut_off.append(int(cell.off_info[0]))
                    if (not np.isnan(cell.off_info[1])):
                        sust_off.append(cell.off_info[1])
                    if (not np.isnan(cell.off_info[2])):
                        delay_off.append(cell.off_info[2])
                if(cell.max_freq_response_off is not None):
                    if (not np.isnan(cell.max_freq_response_off)):
                        max_off.append(int(cell.max_freq_response_off))
            if(((cell.ensembles_on and cell.ensembles_on[number-1] == 1) or (cell.ensembles_off and cell.ensembles_off[number-1]==1)) and not np.isnan(cell.bias_index)):
                bias.append(cell.bias_index)

        if(not cut_on):
            cut_on = [0,0,0,0]
        if(not sust_on):
            sust_on = [0,0,0,0]
        if(not delay_on):
            delay_on = [0,0,0,0]
        if(not max_on):
            max_on = [0,0,0,0]
        if(not cut_off):
            cut_off = [0,0,0,0]
        if(not sust_off):
            sust_off = [0,0,0,0]
        if(not delay_off):
            delay_off = [0,0,0,0]
        if(not max_off):
            max_off = [0,0,0,0]

        cut_on = np.asarray(cut_on)
        sust_on = np.asarray(sust_on)
        delay_on = np.asarray(delay_on)
        cut_off = np.asarray(cut_off)
        sust_off = np.asarray(sust_off)
        delay_off = np.asarray(delay_off)
        max_on = np.asarray(max_on)
        max_off = np.asarray(max_off)

        ax = plt.subplot(gs[0,2])
        n, bins, patches = ax.hist(cut_on, bins = 10, range=[0, 15])
        ax.set_title("freq_stop", fontsize=16)
        mean,std = norm.fit(cut_on)
        xmin, xmax = plt.xlim()
        if std==0:
            std=0.25
            
        x = np.linspace(xmin, xmax, 100)
        y = norm.pdf(x, mean, std)
        l = [max(n)/max(y) * x for x in y]
        ax.plot(x, l)
        ax.set_xlabel('fstimulus [Hz]')

        ax = plt.subplot(gs[1,2])
        n, bins, patches = ax.hist(cut_off, bins = 10, range=[0, 15])
        mean,std = norm.fit(cut_off)
        xmin, xmax = plt.xlim()
        if std==0:
            std=0.25
            
        x = np.linspace(xmin, xmax, 100)
        y = norm.pdf(x, mean, std)
        l = [max(n)/max(y) * x for x in y]
        ax.plot(x, l)
        ax.set_xlabel('fstimulus [Hz]')

        ax = plt.subplot(gs[0,3])
        n, bins, patches = ax.hist(sust_on, bins=10, range=[0, 1],stacked = True)
        ax.set_title("sust_index", fontsize=16)
        mean,std = norm.fit(sust_on)
        xmin, xmax = plt.xlim()
        if std==0:
            std=0.25
            
        x = np.linspace(xmin, xmax, 100)
        y = norm.pdf(x, mean, std)
        l = [max(n)/max(y) * x for x in y]
        ax.plot(x, l)
        ax.set_xlabel('[-]')

        ax = plt.subplot(gs[1,3])
        n, bins, patches = ax.hist(sust_off, bins=10, range=[0, 1], stacked = True)
        mean,std = norm.fit(sust_off)
        xmin, xmax = plt.xlim()
        if std==0:
            std=0.25
            
        x = np.linspace(xmin, xmax, 100)
        y = norm.pdf(x, mean, std)
        l = [max(n)/max(y) * x for x in y]
        ax.plot(x, l)
        ax.set_xlabel('[-]')

        ax = plt.subplot(gs[0,4])
        n, bins, patches = ax.hist(delay_on, bins=10, range=[0, 1])
        ax.set_title("delay", fontsize=16)
        mean,std = norm.fit(delay_on)
        xmin, xmax = plt.xlim()
        if std==0:
            std=0.25
            
        x = np.linspace(xmin, xmax, 100)
        y = norm.pdf(x, mean, std)
        l = [max(n)/max(y) * x for x in y]
        ax.plot(x, l)
        ax.set_xlabel('Delay [s]')
        

        ax = plt.subplot(gs[1,4])
        n, bins, patches = ax.hist(delay_off, bins=10, range=[0, 1])
        mean,std = norm.fit(delay_off)
        xmin, xmax = plt.xlim()
        if std==0:
            std=0.25
            
        x = np.linspace(xmin, xmax, 100)
        y = norm.pdf(x, mean, std)
        l = [max(n)/max(y) * x for x in y]
        ax.plot(x, l)
        ax.set_xlabel('Delay [s]')

        ax = plt.subplot(gs[:,6])
        n, bins, patches = ax.hist(bias, bins = 10, range=[-1, 1])
        ax.set_title("bias_index", fontsize=16)
        mean,std = norm.fit(bias)
        xmin, xmax = plt.xlim()
        if std==0:
            std=0.25
            
        x = np.linspace(xmin, xmax, 100)
        y = norm.pdf(x, mean, std)
        l = [max(n)/max(y) * x for x in y]
        ax.plot(x, l)
        ax.set_xlabel('[-]')

        ax = plt.subplot(gs[:,0])
        ax.text(0.2, 0.5, "Ensemble_{}".format(number), fontsize=20)
        plt.xticks(())
        plt.yticks(())
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax = plt.subplot(gs[0,5])
        n, bins, patches = ax.hist(max_on, bins = 10, range=[0, max(max_on)+1])
        ax.set_title("max_resp", fontsize=16)
        mean,std = norm.fit(max_on)
        xmin, xmax = plt.xlim()
        if std==0:
            std=0.25
            
        x = np.linspace(xmin, xmax, 100)
        y = norm.pdf(x, mean, std)
        l = [max(n)/max(y) * x for x in y]
        ax.plot(x, l)
        ax.set_xlabel('fspike [Hz]')
        

        ax = plt.subplot(gs[1,5])
        n, bins, patches = ax.hist(max_off, bins = 10, range=[0, max(max_off)+1])
        mean,std = norm.fit(max_off)
        xmin, xmax = plt.xlim()
        if std==0:
            std=0.25
            
        x = np.linspace(xmin, xmax, 100)
        y = norm.pdf(x, mean, std)
        l = [max(n)/max(y) * x for x in y]
        ax.plot(x, l)
        ax.set_xlabel('fspike [Hz]')

        ax = plt.subplot(gs[0,1])
        ax.text(0.2, 0.35, "ON", fontsize=20)
        plt.xticks(())
        plt.yticks(())
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax = plt.subplot(gs[1,1])
        ax.text(0.2, 0.35, "OFF", fontsize=20)
        plt.xticks(())
        plt.yticks(())
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig('{}/ensemble_{}_analysis.png'.format(self.cls_folder, number))
        
    def get_bias_histogram(self):
        
        plt.figure(figsize=(16,self.n_cluster*1))

        ward_spike = cluster.hierarchy.linkage(self.spike_dst, method='ward')
        plot_stims = [0,1]
        plot_widths = [1.2, 0.8, 3, 1.5, 1.5, 1.5, 1.5, 1]
        fs=12
        t = np.linspace(-1, 1, 100)
        self.bias_index = np.asarray(self.bias_index)
        gs = gridspec.GridSpec(self.n_cluster, 8, width_ratios = plot_widths, wspace=0.15, hspace=0.5)
        cut_on = []
        sust_on = []
        delay_on = []
        cut_off = []
        sust_off = []
        delay_off = []
        for cell in self.cells:
            if(cell.on_info):
                cut_on.append(cell.on_info[0])
                sust_on.append(cell.on_info[1])
                delay_on.append(cell.on_info[2])
            else:
                cut_on.append(np.nan)
                sust_on.append(np.nan)
                delay_on.append(np.nan)
            if(cell.off_info):
                cut_off.append(cell.off_info[0])
                sust_off.append(cell.off_info[1])
                delay_off.append(cell.off_info[2])
            else:
                cut_off.append(np.nan)
                sust_off.append(np.nan)
                delay_off.append(np.nan)    
        cut_on = np.asarray(cut_on)
        sust_on = np.asarray(sust_on)
        delay_on = np.asarray(delay_on)
        cut_off = np.asarray(cut_off)
        sust_off = np.asarray(sust_off)
        delay_off = np.asarray(delay_off)

        for i in range(1,self.n_cluster +1):
            cls_units = np.asarray(self.exp_unit, dtype='<U9')[self.cls_spike == i]
            cls_chirp = self.chirp_psth[self.cls_spike == i]
            cls_flash = self.flash_psth[self.cls_spike == i]
            cls_bias = self.bias_index[self.cls_spike == i]
            cls_cut_on = cut_on[self.cls_spike == i]
            cls_sust_on = sust_on[self.cls_spike == i]
            cls_delay_on = delay_on[self.cls_spike == i]
            cls_cut_off = cut_off[self.cls_spike == i]
            cls_sust_off = sust_off[self.cls_spike == i]
            cls_delay_off = delay_off[self.cls_spike == i]
            flash_mean = 0
            chirp_mean = 0
            bias_mean = []
            histo_cut_on = []
            histo_delay_on = []
            histo_sust_on = []
            histo_cut_off = []
            histo_delay_off = []
            histo_sust_off = []
            for (uid, chirp, flash, bias, c_on, s_on, d_on, c_off, s_off, d_off) in zip(cls_units, cls_chirp, cls_flash, cls_bias, cls_cut_on, cls_sust_on, cls_delay_on, cls_cut_off, cls_sust_off, cls_delay_off):
                flash_mean += flash
                chirp_mean += chirp
                if (not np.isnan(bias)):
                    bias_mean.append(bias)
                if (not np.isnan(c_on)):
                    histo_cut_on.append(c_on)
                if (not np.isnan(s_on)):
                    histo_sust_on.append(s_on)
                if (not np.isnan(d_on)):
                    histo_delay_on.append(d_on)
                if (not np.isnan(c_off)):
                    histo_cut_off.append(c_off)
                if (not np.isnan(s_off)):
                    histo_sust_off.append(s_off)
                if (not np.isnan(d_off)):
                    histo_delay_off.append(d_off)
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(self.flash_time, flash)
                ax[1].plot(self.chirp_time, chirp)     
                [_ax.clear() for _ax in ax]
                fig.clf()
                fig.clear()
                plt.close()
            flash_mean=flash_mean/len(cls_units)
            chirp_mean=chirp_mean/len(cls_units)

            ax = plt.subplot(gs[(self.n_cluster-(i)),4])
            ax.hist(histo_cut_on, bins = 25, range=[0, 15], color = "royalblue", alpha = 0.75)
            ax.hist(histo_cut_off, bins = 25, range=[0, 15], color = "green", alpha = 0.70)
            x, y = get_fit(histo_cut_on)
            x1, y1 = get_fit(histo_cut_off)
            ax.plot(x, y, color="royalblue")
            ax.plot(x1, y1, color ="green")
            if(i == self.n_cluster):
                ax.set_title("freq_stop", fontsize=16)
            #plt.xticks(())
            
            plt.yticks(())
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax = plt.subplot(gs[(self.n_cluster-(i)),5])
            ax.hist(histo_sust_on, bins = 25, range=[0, 1], color="royalblue", alpha = 0.75)
            ax.hist(histo_sust_off, bins = 25, range=[0, 1], color="green", alpha = 0.70)
            x, y = get_fit(histo_sust_on)
            x1, y1 = get_fit(histo_sust_off)
            ax.plot(x, y, color="royalblue")
            ax.plot(x1, y1, color ="green")
            if(i == self.n_cluster):
                ax.set_title("sust_index", fontsize=16)
            #plt.xticks(())
            plt.yticks(())
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax = plt.subplot(gs[(self.n_cluster-(i)),6])
            ax.hist(histo_delay_on, bins = 25, range=[0, 1], color = "royalblue", alpha = 0.75)
            ax.hist(histo_delay_off, bins = 25, range=[0, 1], color = "green", alpha = 0.70)
            x, y = get_fit(histo_delay_on)
            x1, y1 = get_fit(histo_delay_off)
            ax.plot(x, y, color="royalblue")
            ax.plot(x1, y1, color ="green")
            if(i == self.n_cluster):
                ax.set_title("freq_delay", fontsize=16)
            #plt.xticks(())
            plt.yticks(())
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax = plt.subplot(gs[(self.n_cluster-(i)),3])
            ax.hist(bias_mean, bins = 25, range=[-1, 1])
            mean,std = norm.fit(bias_mean)
            xmin, xmax = plt.xlim()
            if std==0:
                std=0.25
            
            x = np.linspace(xmin, xmax, 100)
            y = norm.pdf(x, mean, std)
            ax.plot(x, y)

            if(i == self.n_cluster):
                ax.set_title("BIAS Histogram", fontsize=16)
            #plt.xticks(())
            
            plt.yticks(())
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax = plt.subplot(gs[(self.n_cluster-(i)),2])
            ax.plot(self.chirp_time, chirp_mean)
            if(i == self.n_cluster):
                ax.set_title("Chirp Response", fontsize=16)
            plt.xticks(())
            plt.yticks(())
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax = plt.subplot(gs[(self.n_cluster-(i)),1])
            ax.plot(self.flash_time, flash_mean)
            if(i == self.n_cluster):
                ax.set_title("Flash Response", fontsize=16)
            plt.xticks(())
            plt.yticks(())
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax = plt.subplot(gs[(self.n_cluster-(i)),7])
            if(i == self.n_cluster):
                ax.set_title("Cell %", fontsize=16)
            ax.text(0.2, 0.35, "{:.2f}%".format(100*(len(cls_units)/len(self.exp_unit))) , fontsize=20)
            plt.xticks(())
            plt.yticks(())
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
        ax = plt.subplot(gs[:,0])
        p = ax.get_position()
        p.x1 = p.x1-0.02
        ax.set_position(p)
        with plt.rc_context({'lines.linewidth': 2, 'font.size':fs, 'axes.titleweight': 'bold'}):
            dend = cluster.hierarchy.dendrogram(ward_spike, p=np.unique(self.cls_spike).shape[0], distance_sort='ascending',
                                        truncate_mode='lastp',orientation='left')

            
            ax.set_xticks(())
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
        plt.savefig('{}/dend_flash_chirp.png'.format(self.cls_folder))
        return plt.gcf()

    def save_data(self):
        np.save('{}/exp_id.npy'.format(self.cls_folder), np.array(self.exp_unit))
        np.save('{}/clust_result.npy'.format(self.cls_folder), np.array(self.cls_spike))
        np.save('{}/chirp_psth.npy'.format(self.cls_folder), self.chirp_psth)
        np.save('{}/flash_psth.npy'.format(self.cls_folder), self.flash_psth)   

        for i in range(1, self.n_cluster + 1):
            if os.path.isdir('{}/{}'.format(self.cls_folder, i)) == False:
                os.mkdir('{}/{}'.format(self.cls_folder, i))
            fig_dir = os.path.join(self.cls_folder, str(i), 'fig')
            if os.path.isdir(fig_dir) == False:
                os.mkdir(fig_dir)
            flash_dir = os.path.join(fig_dir, 'flashes')
            if os.path.isdir(flash_dir) == False:
                os.mkdir(flash_dir)
            freq_dir = os.path.join(fig_dir, 'freq_mod')
            if os.path.isdir(freq_dir) == False:
                os.mkdir(freq_dir)
            amp_dir = os.path.join(fig_dir, 'amp_mod')
            if os.path.isdir(amp_dir) == False:
                os.mkdir(amp_dir)

            # Now resp_dir is vector folder
            resp_dir = [flash_dir, freq_dir, amp_dir]
            resp_file = os.path.join(self.cls_folder, 'response.hdf5')
            feat_file = os.path.join(self.cls_folder, 'features.csv') 
            feat_dir = os.path.join(fig_dir, 'feat')
            if os.path.isdir(feat_dir) == False:
                os.mkdir(feat_dir)

            
            cls_units = np.asarray(self.exp_unit, dtype='<U9')[self.cls_spike == i]
            cls_chirp = self.chirp_psth[self.cls_spike == i]
            cls_flash = self.flash_psth[self.cls_spike == i]
            cls_cells = np.asarray(self.cells)[self.cls_spike == i]

            for (uid, chirp, flash, cell) in zip(cls_units, cls_chirp, cls_flash, cls_cells):
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(self.flash_time, flash)
                ax[1].plot(self.chirp_time, chirp)

                ax[0].set_title('{}, {} flash'.format(uid[0], uid[1]))
                ax[1].set_title('{}, {} chirp'.format(uid[0], uid[1]))
                cell.get_analysis(resp_file, feat_file, resp_dir, feat_dir)
                fig.savefig('{}/{}/{} {}.png'.format(self.cls_folder, i, uid[0], uid[1]))
                
                [_ax.clear() for _ax in ax]
                fig.clf()
                fig.clear()
                plt.close()
                        
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(self.flash_time, np.mean(cls_flash, axis=0))
            ax[1].plot(self.chirp_time, np.mean(cls_chirp, axis=0))
            ax[0].set_title('Mean flash')
            ax[1].set_title('Mean chirp')
            
            fig.savefig('{}/{}/clust{}_mean_resp.png'.format(self.cls_folder, i, i))
            
            [_ax.clear() for _ax in ax]
            fig.clf()
            fig.clear()
            plt.close()

    def get_tsne(self):
        fig, ax = plt.subplots(figsize=(4,2), dpi = 80)

        model = TSNE(n_components=2, random_state=0, perplexity=30)#,init='pca')
        proj = model.fit_transform(self.chirp_psth)

        n_flat_clusters = np.unique(self.cls_spike).shape[0]
        show_order = np.unique(self.cls_spike)[::-1] - 1

        ax.scatter(proj[:, 0], proj[:, 1], c=show_order[self.cls_spike - 1], cmap=ListedColormap(sns.hls_palette(n_flat_clusters, l=0.6, s=0.6).as_hex()))
        fig.savefig('{}/tsne.png'.format(self.cls_folder))
        return fig

    def get_dendogram(self):
        # Mutal Info
        sqr_spike_dst = squareform(self.spike_dst)
        sqr_isi_dst = squareform(self.isi_dst)

        ward_spike = cluster.hierarchy.linkage(self.spike_dst, method='ward')
        ward_isi = cluster.hierarchy.linkage(self.isi_dst, method='ward')

        min_qclust = 6
        max_qclust = 60

        ncls = np.arange(min_qclust, max_qclust, 1)
        metric_scores = np.zeros(ncls.shape[0])
        for i, t in enumerate(ncls):
            self.cls_spike = cluster.hierarchy.fcluster(ward_spike, t=t, criterion='maxclust')
            cls_isi = cluster.hierarchy.fcluster(ward_isi, t=t, criterion='maxclust')
            metric_scores[i] = metrics.adjusted_mutual_info_score(cls_isi, self.cls_spike)
            
        max_cls = np.argmax(metric_scores) + min_qclust
        fig_mut, ax_mut = plt.subplots(figsize=(4,2), dpi = 80)
        ax_mut.plot(ncls, metric_scores)

        ax_mut.axvline(max_cls, ymin=0, ymax=(np.max(metric_scores) - ax_mut.get_ylim()[0]) / np.diff(ax_mut.get_ylim()),
                linestyle='--', color='gray')
        if max_cls not in list(ax_mut.get_xticks()):
            ax_mut.set_xticks(list(ax_mut.get_xticks()) + [max_cls])
        print('Optimum clusters: {}'.format(max_cls))
        fig_mut.savefig('{}/mutual_inf.png'.format(self.cls_folder))

        # Dendogram
        self.n_cluster = max_cls
        self.cls_spike = cluster.hierarchy.fcluster(ward_spike, t=self.n_cluster, criterion='maxclust')
        fig = plt.figure(figsize=(9,4), dpi = 80)
        dn = cluster.hierarchy.dendrogram(ward_spike, p=np.unique(self.cls_spike).shape[0], distance_sort='ascending',
                                        truncate_mode='lastp')
        plt.tight_layout()
        fig.savefig('{}/dendogram.png'.format(self.cls_folder))
        return fig, fig_mut
    
    def read_cell(self, arg):

        cell = Cell(arg[0], arg[1], arg[2], arg[3], arg[9], arg[10])
        cell.set_chirp_response(arg[4])
        cell.set_flash_response(arg[5])
        cell.set_color_response(arg[7], arg[6], arg[8])
        cell.check_quality()
        cell.set_trial_response(self.chirp_time, arg[7])
        cell.get_max_response(arg[11])
        chirp_stimuli = chirp_generator(1/60, chirp_def_args())
        stimuli = chirp_stimuli["full_signal"]
        cls_folder_aux = self.cls_folder
        gs = get_cell_trials(cell.chirp_trials_psth, cell.flash_trials_psth, self.chirp_time, self.flash_time, cell.chirp_psth, cell.flash_psth, cell.exp_unit[0], cell.exp_unit[1], stimuli, arg[1], self.cls_folder)
        self.cls_folder = cls_folder_aux
        return cell
    
    def set_folder(self):
        if os.path.isdir(self.params['Output']) == False:
            os.mkdir(self.params['Output'])
        else:
            shutil.rmtree(self.params['Output'])
            os.mkdir(self.params['Output'])
        self.cls_folder = '{}'.format(self.params['Output'])
        
    def compute_spike_distance(self):
        preload = False
        num_threads = mp.cpu_count()
        # Generate spike distance or load it
        if os.path.isfile('spike_dist.npy') and preload:
            print('Loading spike dist...')
            spike_dst = np.load('{}spike_dist.npy'.format(self.params['Output']))
            print('Done!')
        else:
            flat_units = flat_unit_pairs(self.spiketimes)
            with mp.Pool(num_threads) as pool:
                spike_dst_args = [(self.spiketimes[pair[0]], self.spiketimes[pair[1]]) for pair in flat_units]
                spike_dst = list(tqdm(pool.imap(compute_SPIKE_on_flat_pair, spike_dst_args), total=len(spike_dst_args)))
                self.spike_dst = np.asarray(spike_dst)
            np.save('{}/spike_dist.npy'.format(self.cls_folder), spike_dst)

        # Generate isi distance or load it
        if os.path.isfile('isi_dist.npy') and preload:
            print('Loading isi dist...')
            isi_dst = np.load('{}isi_dist.npy'.format(self.params['Output']))
            print('Done!')
        else:
            flat_units = flat_unit_pairs(self.spiketimes)
            with mp.Pool(num_threads) as pool:
                isi_dst_args = [(self.spiketimes[pair[0]], self.spiketimes[pair[1]]) for pair in flat_units]
                isi_dst = list(tqdm(pool.imap(compute_ISI_on_flat_pair, isi_dst_args), total=len(isi_dst_args)))
                self.isi_dst = np.asarray(isi_dst) 
            np.save('{}/isi_dist.npy'.format(self.cls_folder), isi_dst)
        
    def get_data(self, exps):
        
        ctype_map = {'ON': 0, 'OFF': 1, 'ON/OFF': 2, 'Null': 3}

        chirp_stimuli = chirp_generator(1/60, chirp_def_args())
        stimuli = chirp_stimuli["full_signal"]
        rootdir = os.getcwd()
        num_threads = mp.cpu_count()
        for exp_paths in exps:
            
            os.chdir(exp_paths)
            cfg_file = 'config.ini'
            config.read(cfg_file)
            config.set('PROJECT', 'path', '{}/'.format(os.getcwd()))
            with open(cfg_file, 'w+') as configfile:
                config.write(configfile)
            config.read(cfg_file)

            os.chdir(rootdir)
            exp_path = config['PROJECT']['path']
            exp = config['EXP']['name']
            sorting_file = config['FILES']['sorting']
            sync_file = config['SYNC']['events']
            start_end_path = config['SYNC']['frametime']
            repeated_path = config['SYNC']['repeated']
            sub_events_path = os.path.join(config['SYNC']['folder'],
                                    'sub_event_list_{}_chirp.csv'.format(exp))
            output_features = os.path.join(config['SYNC']['folder'],
                                            'chirp_features_{}.csv'.format(exp))

            sorting = config['FILES']['sorting']

            # names = ['ON', 'OFF', 'adap_0', 'FREQ', 'FREQ_FAST','adap_1', 'AMP', 'adap_2']
            # times = [3, 3, 2, 5, 10, 2, 8, 2] # WARNING: freq time modded
            names = ['ON', 'OFF', 'adap_0', 'FREQ','adap_1', 'AMP', 'adap_2']
            times = [3, 3, 2, 15, 2, 8, 2] # WARNING: freq time modded
            sr = 20000.0 / 1000.0  
            fields_df = ['start_event', 'end_event']
            cfields_df = ['start_event', 'start_next_event']
                
            #Colors
            events = pd.read_csv(sync_file)  
            col_mask = events['protocol_name'] == 'flash'

            blue_mask = col_mask & (events['extra_description'] == 'blue')
            green_mask = col_mask & (events['extra_description'] == 'green')
            blue_time = np.array(events[blue_mask][cfields_df]) / sr
            green_time = np.array(events[green_mask][cfields_df]) / sr
            if(not np.any(blue_time)):
                blue_dur = 0
            else:
                blue_dur = np.max(np.diff(blue_time[:-1], axis=1))
                green_dur = np.max(np.diff(green_time[:-1], axis=1))
                color_dur = blue_dur + green_dur
                    
                blue_time[-1][1] = blue_time[-1][0] + blue_dur
                green_time[-1][1] = green_time[-1][0] + green_dur
                
            events = get_chirp_subevents(sync_file, start_end_path, repeated_path, sub_events_path, names, times)
            if isinstance(events, pd.DataFrame) is not True:
                print('Error computing chirp sub events')
                continue
                
            chirp_mask = events['protocol_name'] == 'chirp'

            mask_on = (events['extra_description'] == 'ON') & chirp_mask
            mask_off = (events['extra_description'] == 'OFF') & chirp_mask
            mask_freq = (events['extra_description'] == 'FREQ') & chirp_mask
            mask_adap1 = (events['extra_description'] == 'adap_1') & chirp_mask
            mask_amp = (events['extra_description'] == 'AMP') & chirp_mask
            mask_adap2 = (events['extra_description'] == 'adap_2') & chirp_mask # Using as final chirp times and for amplitude mod

            on_time = np.array(events[mask_on][fields_df]) / sr
            off_time = np.array(events[mask_off][fields_df]) / sr
            freq_time = np.array(events[mask_freq][fields_df]) / sr
            adap1_time = np.array(events[mask_adap1][fields_df]) / sr
            amp_time = np.array(events[mask_amp][fields_df]) / sr
            adap2_time = np.array(events[mask_adap2][fields_df]) / sr # adap times after amplitude modulation


            bound_time = np.array([freq_time[:, 0], adap2_time[:, 1]]).T
            flash_bound = np.array([on_time[:, 0], off_time[:, 1]]).T / 1000

            flash_dur = 6

            chirp_dur = 35 
            psth_bin = 0.06  # In sec
            fit_resolution = 0.001  # In sec
            self.chirp_time = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / fit_resolution)))
            self.flash_time = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / fit_resolution)))
            color_dur_2 = 12
            self.color_time = np.linspace(0, color_dur_2, int(np.ceil((color_dur_2) / fit_resolution)))
            filtered_cells = 0
            if(not np.any(blue_time)):
                self.color_time = self.flash_time
            
            spec_list = list(events[~events['protocol_spec'].duplicated()]['protocol_spec'])
            spec_events = events
            for i, spec in enumerate(spec_list):
                if len(spec_list) == 1:
                    exp_output = os.path.join(self.params['Output'], exp)
                    if os.path.isdir(exp_output) == False:
                        os.mkdir(exp_output)

                    events = spec_events
                    self.cls_folder = exp_output
                else:
                    if spec == np.nan:
                        spec = i
                    exp_output = os.path.join(self.params['Output'], exp, spec)
                    if os.path.isdir(exp_output) == False:
                        os.makedirs(exp_output)

                    events = spec_events[spec_events['protocol_spec'] == spec]

                    print('Working on spec {} ...'.format(spec))
            
            resp_file = os.path.join(exp_output, 'response.hdf5')
            feat_file = os.path.join(exp_output, 'features.csv') 
            
            with h5py.File(sorting, 'r') as spks:
                idxs = list(spks['spiketimes'].keys())
                uidx = ['Unit_{:04d}'.format(int(i.split('_')[1]) + 1) for i in idxs]
                
                nspikes = {}
                for s in spks['spiketimes']:
                    nspikes[s] = spks['spiketimes'][s][:]
                
                print('{} reading:'.format(exp))
                feat = get_pop_response(nspikes, events, chirp_def_args(), psth_bin, fit_resolution, panalysis=resp_file, feat_file=feat_file)
                feat['exp'] = exp        
                if self.pcells is None: self.pcells = feat.reset_index()
                else: self.pcells = self.pcells.append(feat.reset_index(), ignore_index=True)
                
                cell_args = []
                cell_aux = []
                    
                with mp.Pool(num_threads) as pool:
                    for i, idx in enumerate(idxs):
                        name = (exp, uidx[i], idx)
                        indexer = ctype_map[feat[feat.index == uidx[i]].flash_type[0]]
                        quality = feat['QI'][uidx[i]]                   
                        bias = feat["bias_idx"][uidx[i]]
                        on_info = []
                        off_info = []
                        if(indexer == 0 or indexer == 2):
                            on_freq_cut = feat["on_freq_fcut"][uidx[i]]
                            on_sust_index = feat["on_sust_index"][uidx[i]]
                            on_delay_on = feat["on_freq_delay"][uidx[i]]
                            on_info.append(on_freq_cut)
                            on_info.append(on_sust_index)
                            on_info.append(on_delay_on)
                        if(indexer == 1 or indexer == 2):
                            off_freq_cut = feat["off_freq_fcut"][uidx[i]]
                            off_sust_index = feat["off_sust_index"][uidx[i]]
                            off_delay_on = feat["off_freq_delay"][uidx[i]]
                            off_info.append(off_freq_cut)
                            off_info.append(off_sust_index)
                            off_info.append(off_delay_on)
                        
                        flash_trials = spkt.get_trials(np.unique(spks['spiketimes'][idx][:].flatten() / sr), on_time[:, 0], off_time[:, 1])
                        chirp_trials = spkt.get_trials(np.unique(spks['spiketimes'][idx][:].flatten() / sr), on_time[:, 0], adap2_time[:, 1])

                        if(not np.any(blue_time)):
                            blue_trials = []
                            green_trials = []
                        else:
                            blue_trials = spkt.get_trials(np.unique(spks['spiketimes'][idx][:].flatten() / sr), blue_time[:, 0], blue_time[:, 1])
                            green_trials = spkt.get_trials(np.unique(spks['spiketimes'][idx][:].flatten() / sr), green_time[:, 0], green_time[:, 1])
                        cell_args.append([name, indexer, quality, bias,  chirp_trials, flash_trials, green_trials, blue_trials, blue_dur, on_info, off_info, resp_file])
                    pool_cells = list(tqdm(pool.imap(self.read_cell, cell_args), total=len(cell_args)))      
                    cell_aux = np.asarray(pool_cells)    
                
                for cell in cell_aux:
                    if(cell.low_spikes):                      
                        filtered_cells += 1
                    else:
                        self.exp_unit.append(cell.exp_unit)
                        self.spiketimes.append(cell.spiketimes)
                        self.chirp_psth.append(cell.chirp_psth)
                        self.flash_psth.append(cell.flash_psth)
                        self.bias_index.append(cell.bias_index)   
                        self.cells.append(cell)     
                print("Done!")
    
                print('{} cells below minimum spikes constraint.'.format(filtered_cells))
                print('{} cells valid for {}\n'.format(len(uidx) - filtered_cells, exp))
            for iterate in self.clust_exp:
                if(iterate[1] == exp_paths):
                    self.ensembles_analysis(iterate)
            self.create_csv()
                

        self.chirp_psth = np.asarray(self.chirp_psth)
        self.flash_psth = np.asarray(self.flash_psth)
        print('Total cells for clustering: {}'.format(len(self.exp_unit)))             


        cols = self.pcells.columns.tolist()
        self.pcells = self.pcells[cols[-1:] + cols[:-1]]
    
    