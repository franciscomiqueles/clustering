from spikelib import spiketools as spkt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from functions import format_to_pyspike, create_subplot_ax, time_to_amp
from extract_features import plot_features, plot_resp
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm

class Cell:

    def __init__(self, exp, indexer, quality, bias, on_info, off_info): #blue_time, green_time
        #Cell Info
        self.exp_unit = exp
        self.spiketimes = None
        self.bias_index = bias
        self.type = indexer
        self.quality = quality
        self.on_info = on_info
        self.off_info = off_info
        #Trials
        self.trials = []
        self.trials_color = []
        self.trials_flash = []
        #Cell response
        self.chirp_trials_psth = []
        self.flash_trials_psth = []   
        self.chirp_psth = None
        self.flash_psth = None
        self.color_psth = None
        #If cell is not usable for clustering
        self.low_spikes = False
        self.max_freq_response_on = None
        self.max_freq_response_off = None
        self.ensembles_on = []
        self.ensembles_off = []

    def set_cell_analysis(self, peaks_on, peaks_off, resp_info):
        
        if self.type == 'ON' or self.type == 'ON/OFF':            
            self.peaks_on = peaks_on

        if self.type == 'OFF' or self.type == 'ON/OFF':
            self.peaks_off = peaks_off
        
        self.resp_info = resp_info
            
    def get_analysis(self, resp_file, feat_file, resp_dir, feat_dir):
        with h5py.File(resp_file, 'r') as resp:
            plot_resp(self.exp_unit[1], resp, feat_file, resp_dir)
            plot_features(self.exp_unit[1], resp, feat_file, feat_dir)
            plt.close('all')
    
    def get_max_response(self, resp_file):
        with h5py.File(resp_file, 'r') as resp:
            data = resp['{}/cell_resp/'.format(self.exp_unit[1])]
            if self.type == 0 or self.type == 2:
                t = data['freq_on_peaks'][:][0]
                v = data['freq_on_peaks'][:][1]
                self.max_freq_response_on = max(v)
            if self.type == 1 or self.type == 2:
                t = data['freq_off_peaks'][:][0]
                v = data['freq_off_peaks'][:][1]
                self.max_freq_response_off = max(v)

    def get_ensembles(self, file, n, it):
        if(self.type == 0 or self.type == 2):
            df_feat = pd.read_csv(file, index_col=0)
            info = df_feat.loc[df_feat['unit'] == self.exp_unit[1]]
            if(not info.empty):
                for i in range(1, n+1):
                    if(self.type == 2):
                        info = info.loc[info["flash_type"] == "ON"]
                    self.ensembles_on.append(info.loc[:,"Ensemble_{}".format(i)].item())
                print("EXP: {} has ensembles {}".format(self.exp_unit[1], self.ensembles_on))
                print(it)
        if(self.type == 1 or self.type == 2):
            df_feat = pd.read_csv(file, index_col=0)
            info = df_feat.loc[df_feat['unit'] == self.exp_unit[1]]
            if(not info.empty):
                for i in range(1, n+1):
                    if(self.type == 2):
                        info = info.loc[info["flash_type"] == "OFF"]
                    self.ensembles_off.append(info.loc[:,"Ensemble_{}".format(i)].item())
                print("EXP: {} has ensembles {}".format(self.exp_unit[1], self.ensembles_off))
                print(it)
        it= it+1
        return(it)
                


    def set_chirp_response(self, chirp_trials):
        chirp_dur = 35 
        psth_bin = 0.06  # In sec
        fit_resolution = 0.001  # In sec

        chirp_time = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / fit_resolution)))
        chirp_bins = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / psth_bin)))
        sec_trials = []
        
        for (c) in  chirp_trials:
            t = c
            self.trials.append(t)
            sec_trials.append(t / 1000)
                        
            m = np.zeros_like(t, dtype=bool)
            m[np.unique(t, return_index=True)[1]] = True
            if t[~m].shape[0] > 0:
                print(t[~m])
    
        spikes_chirp = spkt.flatten_trials(sec_trials)
        (psth, _) = np.histogram(spikes_chirp, bins=chirp_bins)
        self.chirp_psth = spkt.est_pdf(sec_trials, chirp_time, bandwidth=psth_bin, norm_factor=psth.max())
        self.spiketimes = np.asarray(format_to_pyspike(self.trials, (chirp_dur) * 1000), dtype=object)

    def set_flash_response(self, flash_trials):
        flash_dur = 6
        psth_bin = 0.06  # In sec
        fit_resolution = 0.001  # In sec

        flash_time = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / fit_resolution)))
        flash_bins = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / psth_bin)))
        sec_trials = []
        for (c) in  flash_trials:
            t = c
            self.trials_flash.append(t)
            sec_trials.append(t / 1000)
                        
            m = np.zeros_like(t, dtype=bool)
            m[np.unique(t, return_index=True)[1]] = True
            if t[~m].shape[0] > 0:
                print(t[~m])

        spikes_flash = spkt.flatten_trials(sec_trials)
        (psth, _) = np.histogram(spikes_flash , bins=flash_bins)
        self.flash_psth = spkt.est_pdf(sec_trials, flash_time, bandwidth=psth_bin, norm_factor=psth.max())

    def set_color_response(self, blue_trials, green_trials, blue_dur):
        
        
        psth_bin = 0.06  # In sec
        fit_resolution = 0.001  # In sec
        color_dur_2 = 12
        color_time = np.linspace(0, color_dur_2, int(np.ceil((color_dur_2) / fit_resolution)))
        color_bins = np.linspace(0, color_dur_2, int(np.ceil((color_dur_2) / psth_bin)))


        sec_trials_color = []
        if(blue_trials):
            for (b, g) in zip(blue_trials, green_trials):
                t_color = np.concatenate([b, g + blue_dur])
                self.trials_color.append(t_color)
                sec_trials_color.append(t_color / 1000)

            spikes_color = spkt.flatten_trials(self.trials_color)
            (psth, _) = np.histogram(spikes_color / 1000, bins=color_bins)
            self.color_psth = spkt.est_pdf(sec_trials_color , color_time, bandwidth=psth_bin, norm_factor=psth.max())
        else:
            self.color_psth = self.flash_psth

    def set_trial_response(self, chirp_time, blue_trials):

        psth_bin = 0.06  # In sec
        fit_resolution = 0.001  # In sec
        color_dur_2 = 12
        chirp_dur = 35 
        flash_dur = 6
        color_time = np.linspace(0, color_dur_2, int(np.ceil((color_dur_2) / fit_resolution)))
        color_bins = np.linspace(0, color_dur_2, int(np.ceil((color_dur_2) / psth_bin)))
        flash_time = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / fit_resolution)))
        
        chirp_time = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / fit_resolution)))
        chirp_bins = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / psth_bin)))
        for j in range(len(self.trials)):               
            (psth, _) = np.histogram(self.trials[j] / 1000, bins=chirp_bins)
            chirp_resp = spkt.est_pdf([self.trials[j]/ 1000], chirp_time, bandwidth=psth_bin, norm_factor=psth.max())                   
            self.chirp_trials_psth.append(chirp_resp)
            if(blue_trials):
                (psth, _) = np.histogram(self.trials_color[j]/1000, bins=color_bins)
                flash_resp = spkt.est_pdf([self.trials_color[j]/1000], self.clust.color_time, bandwidth=psth_bin, norm_factor=psth.max())
                self.flash_trials_psth.append(flash_resp)
            else:
                (psth, _) = np.histogram(self.trials_flash[j]/1000, bins=color_bins)
                flash_resp = spkt.est_pdf([self.trials_flash[j]/1000], flash_time, bandwidth=psth_bin, norm_factor=psth.max())
                self.flash_trials_psth.append(flash_resp)

    def check_quality(self):

        if self.quality < 0.001:
            self.low_spikes = True
        
        for t in self.trials:

            if t.shape[0] < 1:
                self.low_spikes = True
            break 
  
    def get_cell_figure(self, cls_folder, stimuli):
        fig = plt.figure(figsize=(16,20))

        chirp_dur = 35 
        psth_bin = 0.06  # In sec
        fit_resolution = 0.001  # In sec
        chirp_time = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / fit_resolution)))
        
        color_dur_2 = 12 
        color_time = np.linspace(0, color_dur_2, int(np.ceil((color_dur_2) / fit_resolution)))
        plot_stims = [0,1]
        plot_widths = [1.2,0.8,3]
    
        m_c = np.max(self.chirp_trials_psth) + 5
        m_f = np.max(self.flash_trials_psth) + 5
    
        gs = gridspec.GridSpec(23, 3, width_ratios = plot_widths, wspace=0.1, hspace=0.1)
        
        if(self.indexer == 0):
            if os.path.isdir('{}/ON/{}/'.format(cls_folder, self.exp_unit[0])) == False:
                os.mkdir('{}/ON/{}/'.format(cls_folder, self.exp_unit[0]))
            cls_folder = '{}/ON/'.format(cls_folder)
        
        if(self.indexer == 1):
            if os.path.isdir('{}/OFF/{}/'.format(cls_folder, self.exp_unit[0])) == False:
                os.mkdir('{}/OFF/{}/'.format(cls_folder, self.exp_unit[0]))
            cls_folder = '{}/OFF/'.format(cls_folder)
            
        if(self.indexer == 2):
            if os.path.isdir('{}/{}/'.format(cls_folder, self.exp_unit[0])) == False:
                os.mkdir('{}/{}/'.format(cls_folder, self.exp_unit[0]))
            cls_folder = '{}/ON-OFF/'.format(cls_folder)
            
        if(self.indexer == 3):
            if os.path.isdir('{}/{}/'.format(cls_folder, self.exp_unit[0])) == False:
                os.mkdir('{}/Null/{}/'.format(cls_folder, self.exp_unit[0]))
            cls_folder = '{}/Null/'.format(cls_folder)

        for i in range(len(self.chirp_trials_psth)):
            if(i==0):
                ax = create_subplot_ax(gs[i, 2], [chirp_time, self.chirp_trials_psth[i]], False, False, ylim = [0 , m_c], title = "Chirp Response")
            else:
                ax = create_subplot_ax(gs[i, 2], [chirp_time, self.chirp_trials_psth[i]], False, False,ylim = [0 , m_c])
            if(i==0):
                ax = create_subplot_ax(gs[i, 1], [color_time, self.flash_trials_psth[i]], False, False, ylim = [0 , m_f], title = "Color Response")
            else:
                ax = create_subplot_ax(gs[i, 1], [color_time, self.flash_trials_psth[i]], False, False, ylim = [0 , m_f])
            
            ax = create_subplot_ax(gs[i, 0], visible = False, title = "Trial {}".format(i+1), y = 0.35)
        
        ax = create_subplot_ax(gs[21, 1], [color_time, self.color_psth], False, False)
        ax = create_subplot_ax(gs[21, 2], [chirp_time, self.chirp_psth], False, False)
        ax = create_subplot_ax(gs[21, 0], visible = False, title = "Final Psth", y = 0.35)
        ax = create_subplot_ax(gs[22, 0], visible = False, title = "Stimuli", y = 0.35)
        ax = create_subplot_ax(gs[22, 2], stimuli, False) 
        ax = create_subplot_ax(gs[22, 1], np.concatenate([stimuli[0:360], stimuli[0:360]]), False)
                
        plt.savefig('{}/{}/{}.png'.format(cls_folder, self.exp_unit[0], self.exp_unit[1]))
        fig.clf()
        plt.close(fig)

