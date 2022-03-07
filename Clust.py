from __future__ import absolute_import, print_function

from configparser import ConfigParser, ExtendedInterpolation
                                                                                                                                                                                
import scipy.io
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
from functions import format_to_pyspike, get_cell_trials, compute_ISI_on_flat_pair, compute_SPIKE_on_flat_pair, flat_unit_pairs, format_to_pyspike
from Cell import Cell
import json

config = ConfigParser(interpolation=ExtendedInterpolation())


class Clust:

    def __init__(self, folder):

        self.exp_unit = []
        self.spiketimes = []
        self.chirp_psth = []
        self.flash_psth = []
        self.bias_index = []

        self.pcells = None
        self.params = None

        self.cls_folder = folder

        self.spike_dst = None
        self.isi_dst = None

        self.n_cluster = None
        self.cls_spike = None

    def read_params(self):
        with open("params.json") as p:
            self.params = json.load(p)
    
        # Create output folder if it does not exist
        if os.path.isdir(self.params['Output']) == False:
            os.mkdir(self.params['Output'])
        
    def get_bias_histogram(self):
        
        chirp_dur = 35 
        psth_bin = 0.06  # In sec
        fit_resolution = 0.001  # In sec
        chirp_time = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / fit_resolution)))

        flash_dur = 6
        flash_time = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / fit_resolution)))

        plt.figure(figsize=(16,self.n_cluster*1))

        ward_spike = cluster.hierarchy.linkage(self.spike_dst, method='ward')
        plot_stims = [0,1]
        plot_widths = [1.2, 0.8, 3, 1.5, 1]
        fs=12
        t = np.linspace(-1, 1, 100)
        self.bias_index = np.asarray(self.bias_index)
        gs = gridspec.GridSpec(self.n_cluster, 5, width_ratios = plot_widths, wspace=0.15, hspace=0.5)
        for i in range(1,self.n_cluster +1):
            cls_units = np.asarray(self.exp_unit, dtype='<U9')[self.cls_spike == i]
            cls_chirp = self.chirp_psth[self.cls_spike == i]
            cls_flash = self.flash_psth[self.cls_spike == i]
            cls_bias = self.bias_index[self.cls_spike == i]
            flash_mean = 0
            chirp_mean = 0
            bias_mean = []
            bias_mean_aux = []
            for (uid, chirp, flash, bias) in zip(cls_units, cls_chirp, cls_flash, cls_bias):
                flash_mean += flash
                chirp_mean += chirp
                if (not np.isnan(bias)):
                    bias_mean.append(bias)
                    bias_mean_aux.append(bias / 1000)
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(flash_time, flash)
                ax[1].plot(chirp_time, chirp)     
                [_ax.clear() for _ax in ax]
                fig.clf()
                fig.clear()
                plt.close()
            flash_mean=flash_mean/len(cls_units)
            chirp_mean=chirp_mean/len(cls_units)
            
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
            ax.plot(chirp_time, chirp_mean)
            if(i == self.n_cluster):
                ax.set_title("Chirp Response", fontsize=16)
            plt.xticks(())
            plt.yticks(())
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax = plt.subplot(gs[(self.n_cluster-(i)),1])
            ax.plot(flash_time, flash_mean)
            if(i == self.n_cluster):
                ax.set_title("Flash Response", fontsize=16)
            plt.xticks(())
            plt.yticks(())
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax = plt.subplot(gs[(self.n_cluster-(i)),4])
            if(i == self.n_cluster):
                ax.set_title("Cell Percentage", fontsize=16)
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

    def save_data(self):
        chirp_dur = 35 
        psth_bin = 0.06  # In sec
        fit_resolution = 0.001  # In sec
        chirp_time = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / fit_resolution)))
        
        flash_dur = 6
        flash_time = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / fit_resolution)))

        np.save('{}/exp_id.npy'.format(self.cls_folder), np.array(self.exp_unit))
        np.save('{}/clust_result.npy'.format(self.cls_folder), np.array(self.cls_spike))
        np.save('{}/chirp_psth.npy'.format(self.cls_folder), self.chirp_psth)
        np.save('{}/flash_psth.npy'.format(self.cls_folder), self.flash_psth)   

        for i in range(1, self.n_cluster + 1):
            if os.path.isdir('{}/{}'.format(self.cls_folder, i)) == False:
                os.mkdir('{}/{}'.format(self.cls_folder, i))
            
            cls_units = np.asarray(self.exp_unit, dtype='<U9')[self.cls_spike == i]
            cls_chirp = self.chirp_psth[self.cls_spike == i]
            cls_flash = self.flash_psth[self.cls_spike == i]

            for (uid, chirp, flash) in zip(cls_units, cls_chirp, cls_flash):
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(flash_time, flash)
                ax[1].plot(chirp_time, chirp)

                ax[0].set_title('{}, {} flash'.format(uid[0], uid[1]))
                ax[1].set_title('{}, {} chirp'.format(uid[0], uid[1]))
                
                fig.savefig('{}/{}/{} {}.png'.format(self.cls_folder, i, uid[0], uid[1]))
                
                [_ax.clear() for _ax in ax]
                fig.clf()
                fig.clear()
                plt.close()
                        
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(flash_time, np.mean(cls_flash, axis=0))
            ax[1].plot(chirp_time, np.mean(cls_chirp, axis=0))
            ax[0].set_title('Mean flash')
            ax[1].set_title('Mean chirp')
            
            fig.savefig('{}/{}/clust{}_mean_resp.png'.format(self.cls_folder, i, i))
            
            [_ax.clear() for _ax in ax]
            fig.clf()
            fig.clear()
            plt.close()

    def get_tsne(self):
        fig, ax = plt.subplots()

        model = TSNE(n_components=2, random_state=0, perplexity=30)#,init='pca')
        proj = model.fit_transform(self.chirp_psth)

        n_flat_clusters = np.unique(self.cls_spike).shape[0]
        show_order = np.unique(self.cls_spike)[::-1] - 1

        ax.scatter(proj[:, 0], proj[:, 1], c=show_order[self.cls_spike - 1], cmap=ListedColormap(sns.hls_palette(n_flat_clusters, l=0.6, s=0.6).as_hex()))
        fig.savefig('{}/tsne.png'.format(self.cls_folder))

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
        fig, ax = plt.subplots()
        ax.plot(ncls, metric_scores)

        ax.axvline(max_cls, ymin=0, ymax=(np.max(metric_scores) - ax.get_ylim()[0]) / np.diff(ax.get_ylim()),
                linestyle='--', color='gray')
        if max_cls not in list(ax.get_xticks()):
            ax.set_xticks(list(ax.get_xticks()) + [max_cls])
        print('Optimum clusters: {}'.format(max_cls))
        fig.savefig('{}/mutual_inf.png'.format(self.cls_folder))

        # Dendogram
        self.n_cluster = max_cls
        self.cls_spike = cluster.hierarchy.fcluster(ward_spike, t=self.n_cluster, criterion='maxclust')
        fig = plt.figure()
        dn = cluster.hierarchy.dendrogram(ward_spike, p=np.unique(self.cls_spike).shape[0], distance_sort='ascending',
                                        truncate_mode='lastp')
        plt.tight_layout()
        fig.savefig('{}/dendogram.png'.format(self.cls_folder))

    def set_folder(self):
        if os.path.isdir(self.params['Output']) == False:
            os.mkdir(self.params['Output'])
        else:
            shutil.rmtree(self.params['Output'])
            os.mkdir(self.params['Output'])
        self.cls_folder = '{}/{}'.format(self.params['Output'], self.cls_folder)
        if os.path.isdir(self.cls_folder) == False:
            os.mkdir(self.cls_folder)
        else:
            shutil.rmtree(self.cls_folder)
            os.mkdir(self.cls_folder)

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
    
    def get_data(self):
        
        ctype_map = {'ON': 0, 'OFF': 1, 'ON/OFF': 2, 'Null': 3}

        chirp_stimuli = chirp_generator(1/60, chirp_def_args())
        stimuli = chirp_stimuli["full_signal"]
        
        rootdir = os.getcwd()
        
        for exp_path in self.params['Experiments']:
            os.chdir(exp_path)
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

            names = ['ON', 'OFF', 'adap_0', 'FREQ', 'FREQ_FAST','adap_1', 'AMP', 'adap_2']
            times = [3, 3, 2, 5, 10, 2, 8, 2] # WARNING: freq time modded
            sr = 20000.0 / 1000.0  
            fields_df = ['start_event', 'end_event']
            cfields_df = ['start_event', 'start_next_event']
            
            #Colors
            # events = pd.read_csv(sync_file)  
            # col_mask = events['protocol_name'] == 'flash'

            # blue_mask = col_mask & (events['extra_description'] == 'blue')
            # green_mask = col_mask & (events['extra_description'] == 'green')
            # print(blue_mask)
            # blue_time = np.array(events[blue_mask][cfields_df]) / sr
            # green_time = np.array(events[green_mask][cfields_df]) / sr
            
            # blue_dur = np.max(np.diff(blue_time[:-1], axis=1))
            # green_dur = np.max(np.diff(green_time[:-1], axis=1))
            # color_dur = blue_dur + green_dur
            
            # blue_time[-1][1] = blue_time[-1][0] + blue_dur
            # green_time[-1][1] = green_time[-1][0] + green_dur
            
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

            psth_bin = 0.06  # In sec
            fit_resolution = 0.001  # In sec

            filtered_cells = 0

            with h5py.File(sorting, 'r') as spks:
                idxs = list(spks['spiketimes'].keys())
                uidx = ['Unit_{:04d}'.format(int(i.split('_')[1]) + 1) for i in idxs]
                
                nspikes = {}
                for s in spks['spiketimes']:
                    nspikes[s] = spks['spiketimes'][s][:]
                
                print('{} reading:'.format(exp))
                feat = get_pop_response(nspikes, events, chirp_def_args(), psth_bin, fit_resolution)
                feat['exp'] = exp        
                if self.pcells is None: self.pcells = feat.reset_index()
                else: self.pcells = self.pcells.append(feat.reset_index(), ignore_index=True)
                
                for i, idx in enumerate(idxs):
                    name = (exp, uidx[i], idx)
                    indexer = ctype_map[feat[feat.index == uidx[i]].flash_type[0]]
                    quality = feat['QI'][uidx[i]]                   
                    bias = feat["bias_idx"][uidx[i]]
                    cell = Cell(name, indexer, quality, bias, spks)
                    cell.set_chirp_response(spks, on_time, adap2_time, sr)
                    cell.set_flash_response(spks, flash_bound, sr)
                    cell.check_quality()
                    if(cell.low_spikes):
                        filtered_cells += 1
                    else:
                        self.exp_unit.append(cell.exp_unit)
                        self.spiketimes.append(cell.spiketimes)
                        self.chirp_psth.append(cell.chirp_psth)
                        self.flash_psth.append(cell.flash_psth)
                        self.bias_index.append(cell.bias_index)
                    
                    # gs = get_cell_trials(chirp_trials_cell, flash_trials_cell, self.chirp_time, color_time, chirp_resp, color_resp, exp, uidx[i], stimuli, indexer, self.cls_folder)
            print('{} cells below minimum spikes constraint.'.format(filtered_cells))
            print('{} cells valid for {}\n'.format(len(uidx) - filtered_cells, exp))

        self.chirp_psth = np.asarray(self.chirp_psth)
        self.flash_psth = np.asarray(self.flash_psth)
        print('Total cells for clustering: {}'.format(len(self.exp_unit)))             


        cols = self.pcells.columns.tolist()
        self.pcells = self.pcells[cols[-1:] + cols[:-1]]
    
    
