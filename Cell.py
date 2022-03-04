from spikelib import spiketools as spkt
from functions import format_to_pyspike
import numpy as np

class Cell:

    def __init__(self, exp, indexer, quality, bias, spks, sr, idx, on_time, adap2_time, flash_bound): #blue_time, green_time

        self.exp_unit = exp
        self.spiketimes = None
        self.bias_index = bias
        self.type = indexer
        self.quality = quality

        self.trials = []
        self.trials_color = []
        
        self.chirp_trials_psth = []
        self.flash_trials_psth = []
        
        self.low_spikes = False
        
        self.chirp_psth = None
        self.flash_psth = None
        self.color_psth = None

    def set_chirp_response(self, spks, idx, on_time, adap2_time, sr):
        chirp_dur = 35 
        psth_bin = 0.06  # In sec
        fit_resolution = 0.001  # In sec

        chirp_time = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / fit_resolution)))
        chirp_bins = np.linspace(0, chirp_dur, int(np.ceil((chirp_dur) / psth_bin)))
        sec_trials = []
        chirp_trials = spkt.get_trials(np.unique(spks['spiketimes'][idx][:].flatten() / sr), on_time[:, 0], adap2_time[:, 1])
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

    def set_flash_response(self, spks, idx, flash_bound, sr):
        flash_dur = 6
        psth_bin = 0.06  # In sec
        fit_resolution = 0.001  # In sec

        flash_time = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / fit_resolution)))
        flash_bins = np.linspace(0, flash_dur, int(np.ceil((flash_dur) / psth_bin)))

        flash_trials = spkt.get_trials(np.unique(spks['spiketimes'][idx][:].flatten() / sr) / 1000, flash_bound[:, 0], flash_bound[:, 1])
        spikes_flash = spkt.flatten_trials(flash_trials)
        (psth, _) = np.histogram(spikes_flash, bins=flash_bins)
        self.flash_psth = spkt.est_pdf(flash_trials, flash_time, bandwidth=psth_bin, norm_factor=psth.max())

    def set_color_response(self, spks, idx, blue_time, green_time, sr):
        #Blue flash
        blue_trials = spkt.get_trials(np.unique(spks['spiketimes'][idx][:].flatten() / sr), blue_time[:, 0], blue_time[:, 1])
        #Green flash
        green_trials = spkt.get_trials(np.unique(spks['spiketimes'][idx][:].flatten() / sr), green_time[:, 0], green_time[:, 1])
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
            #print("chao")
        
        for t in self.trials:

            if t.shape[0] < 20:
                self.low_spikes = True
                #print("hola")
            break 

