from spikelib import spiketools as spkt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from functions import format_to_pyspike, create_subplot_ax
import numpy as np

class Cell:

    def __init__(self, exp, indexer, quality, bias, spks): #blue_time, green_time
        #Cell Info
        self.exp_unit = exp
        self.spiketimes = None
        self.bias_index = bias
        self.type = indexer
        self.quality = quality
        #Trials
        self.trials = []
        self.trials_color = []
        #Cell response
        self.chirp_trials_psth = []
        self.flash_trials_psth = []   
        self.chirp_psth = None
        self.flash_psth = None
        self.color_psth = None
        #If cell is not usable for clustering
        self.low_spikes = False

    def set_chirp_response(self, spks, on_time, adap2_time, sr):
        chirp_dur = 35 
        psth_bin = 0.06  # In sec
        fit_resolution = 0.001  # In sec

        chirp_time = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / fit_resolution)))
        chirp_bins = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / psth_bin)))
        sec_trials = []
        chirp_trials = spkt.get_trials(np.unique(spks['spiketimes'][self.name[2]][:].flatten() / sr), on_time[:, 0], adap2_time[:, 1])
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

    def set_flash_response(self, spks, flash_bound, sr):
        flash_dur = 6
        psth_bin = 0.06  # In sec
        fit_resolution = 0.001  # In sec

        flash_time = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / fit_resolution)))
        flash_bins = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / psth_bin)))

        flash_trials = spkt.get_trials(np.unique(spks['spiketimes'][self.name[2]][:].flatten() / sr) / 1000, flash_bound[:, 0], flash_bound[:, 1])
        spikes_flash = spkt.flatten_trials(flash_trials)
        (psth, _) = np.histogram(spikes_flash, bins=flash_bins)
        self.flash_psth = spkt.est_pdf(flash_trials, flash_time, bandwidth=psth_bin, norm_factor=psth.max())

    def set_color_response(self, spks, blue_time, green_time, sr):
        #Blue flash
        blue_trials = spkt.get_trials(np.unique(spks['spiketimes'][self.name[2]][:].flatten() / sr), blue_time[:, 0], blue_time[:, 1])
        #Green flash
        green_trials = spkt.get_trials(np.unique(spks['spiketimes'][self.name[2]][:].flatten() / sr), green_time[:, 0], green_time[:, 1])
        blue_dur = blue_dur = np.max(np.diff(blue_time[:-1], axis=1))
        
        psth_bin = 0.06  # In sec
        fit_resolution = 0.001  # In sec
        color_dur_2 = 12
        color_time = np.linspace(0, color_dur_2, int(np.ceil((color_dur_2) / fit_resolution)))
        color_bins = np.linspace(0, color_dur_2, int(np.ceil((color_dur_2) / psth_bin)))

        for (b, g) in zip(blue_trials, green_trials):
            t_color = np.concatenate([b, g + blue_dur])
            self.trials_color.append(t_color)
            self.sec_trials_color.append(t_color / 1000)

        spikes_color = spkt.flatten_trials(self.trials_color)
        (psth, _) = np.histogram(spikes_color / 1000, bins=color_bins)
        self.color_psth = spkt.est_pdf(self.sec_trials_color , color_time, bandwidth=psth_bin, norm_factor=psth.max())

    def set_trial_response(self):

        psth_bin = 0.06  # In sec
        fit_resolution = 0.001  # In sec
        color_dur_2 = 12
        chirp_dur = 35 
        color_time = np.linspace(0, color_dur_2, int(np.ceil((color_dur_2) / fit_resolution)))
        color_bins = np.linspace(0, color_dur_2, int(np.ceil((color_dur_2) / psth_bin)))
        
        chirp_time = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / fit_resolution)))
        chirp_bins = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / psth_bin)))
        for j in range(len(self.trials)):               
            (psth, _) = np.histogram(self.trials[j] / 1000, bins=chirp_bins)
            chirp_resp = spkt.est_pdf([self.trials[j]/ 1000], chirp_time, bandwidth=psth_bin, norm_factor=psth.max())                   
            self.chirp_trials_psth.append(chirp_resp)

            (psth, _) = np.histogram(self.trials_color[j]/1000, bins=color_bins)
            flash_resp = spkt.est_pdf([self.trials_color[j]/1000], color_time, bandwidth=psth_bin, norm_factor=psth.max())

            self.flash_trials_psth.append(flash_resp)
    
    def check_quality(self):

        if self.quality < 0.25:
            self.low_spikes = True
        
        for t in self.trials:

            if t.shape[0] < 20:
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
            if os.path.isdir('{}/ON/{}/'.format(cls_folder, self.name[0])) == False:
                os.mkdir('{}/ON/{}/'.format(cls_folder, self.name[0]))
            cls_folder = '{}/ON/'.format(cls_folder)
        
        if(self.indexer == 1):
            if os.path.isdir('{}/OFF/{}/'.format(cls_folder, self.name[0])) == False:
                os.mkdir('{}/OFF/{}/'.format(cls_folder, self.name[0]))
            cls_folder = '{}/OFF/'.format(cls_folder)
            
        if(self.indexer == 2):
            if os.path.isdir('{}/{}/'.format(cls_folder, self.name[0])) == False:
                os.mkdir('{}/{}/'.format(cls_folder, self.name[0]))
            cls_folder = '{}/ON-OFF/'.format(cls_folder)
            
        if(self.indexer == 3):
            if os.path.isdir('{}/{}/'.format(cls_folder, self.name[0])) == False:
                os.mkdir('{}/Null/{}/'.format(cls_folder, self.name[0]))
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
                
        plt.savefig('{}/{}/{}.png'.format(cls_folder, self.name[0], self.name[1]))
        fig.clf()
        plt.close(fig)
        

