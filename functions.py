import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import os
import numpy as np
import pyspike
from scipy.stats import norm

def create_subplot_ax(position, plot = None, visible = True, tick = True,  title = "", xlim = None , ylim = None, y = None ):
    ax = plt.subplot(position)
    if(len(plot) == 1):
        ax.plot(plot)
    if(len(plot) == 2):
        ax.plot(plot[0], plot[1])
    else:
        print("No plot")
    
    if(not visible):
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if(tick):
        plt.xticks(())
        plt.yticks(())
    ax.set_title("{}".format(title), fontsize=20, y = y)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax

def get_cell_trials(chirp_trials, flash_trials, chirp_time, flash_time, chirp_psth, flash_psth, exp, uidx, stimuli, indexer, cls_folder):
    fig = plt.figure(figsize=(16,20))

    plot_stims = [0,1]
    plot_widths = [1.2,0.8,3]
    
    m_c = np.max(chirp_trials) + 5
    m_f = np.max(flash_trials) + 5
    
    gs = gridspec.GridSpec(23, 3, width_ratios = plot_widths, wspace=0.1, hspace=0.1)

    if(indexer == 0):
        if os.path.isdir('{}/ON/'.format(cls_folder)) == False:
            os.mkdir('{}/ON/'.format(cls_folder))
        
        
    if(indexer == 1):
        if os.path.isdir('{}/OFF/'.format(cls_folder)) == False:
            os.mkdir('{}/OFF/'.format(cls_folder))
       
        
    if(indexer == 2):
        if os.path.isdir('{}/ON-OFF/'.format(cls_folder)) == False:
            os.mkdir('{}/ON-OFF/'.format(cls_folder))
       
        
    if(indexer == 3):
        if os.path.isdir('{}/Null/'.format(cls_folder)) == False:
            os.mkdir('{}/Null/'.format(cls_folder))
       

    if(indexer == 0):
        if os.path.isdir('{}/ON/{}/'.format(cls_folder, exp)) == False:
            os.mkdir('{}/ON/{}/'.format(cls_folder, exp))
        cls_folder = '{}/ON/'.format(cls_folder)
        
    if(indexer == 1):
        if os.path.isdir('{}/OFF/{}/'.format(cls_folder, exp)) == False:
            os.mkdir('{}/OFF/{}/'.format(cls_folder, exp))
        cls_folder = '{}/OFF/'.format(cls_folder)
        
    if(indexer == 2):
        if os.path.isdir('{}/ON-OFF/{}/'.format(cls_folder, exp)) == False:
            os.mkdir('{}/ON-OFF/{}/'.format(cls_folder, exp))
        cls_folder = '{}/ON-OFF/'.format(cls_folder)
        
    if(indexer == 3):
        if os.path.isdir('{}/Null/{}/'.format(cls_folder, exp)) == False:
            os.mkdir('{}/Null/{}/'.format(cls_folder, exp))
        cls_folder = '{}/Null/'.format(cls_folder)

    
        
        
        
    for i in range(len(chirp_trials)):
        ax = plt.subplot(gs[i , 2])
        ax.plot(chirp_time, chirp_trials[i])

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0 , m_c])
        
        if(i==0):
            ax.set_title("Chirp Response", fontsize=20)
        
        
        ax = plt.subplot(gs[i , 1])
        ax.plot(flash_time, flash_trials[i])

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0 , m_f])
        
        if(i==0):
            ax.set_title("Color Response", fontsize=20)
        
        ax = plt.subplot(gs[i , 0])
        ax.set_title("Trial {}".format(i+1), y = 0.35)
        plt.xticks(())
        plt.yticks(())
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    ax = plt.subplot(gs[21, 1])
    ax.plot(flash_time, flash_psth)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax = plt.subplot(gs[21 , 2])
    ax.plot(chirp_time, chirp_psth)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax = plt.subplot(gs[21 , 0])
    ax.set_title("Final PSTH", y = 0.35)
    plt.xticks(())
    plt.yticks(())
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax = plt.subplot(gs[22 , 0])
    ax.set_title("Stimuli", y = 0.35)
    plt.xticks(())
    plt.yticks(())
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax = plt.subplot(gs[22 , 2])
    ax.plot(stimuli)
    plt.xticks(())
    plt.yticks(())
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax = plt.subplot(gs[22 , 1])
    ax.plot(np.concatenate([stimuli[0:360], stimuli[0:360]]))
    plt.xticks(())
    plt.yticks(())
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
            
    plt.savefig('{}/{}/{}.png'.format(cls_folder, exp, uidx))
    fig.clf()
    plt.close(fig)
    
    return gs

def flat_unit_pairs(st):
    
    flat_pairs = []
    for i in range(len(st)):
        for j in range(i + 1, len(st)):
            flat_pairs.append((i, j))
    return flat_pairs

def compute_SPIKE_on_flat_pair(arg):

    sti, stj = arg
    assert sti.shape[0] == stj.shape[0]
    ds = []
    for i in range(sti.shape[0]):
        for j in range(stj.shape[0]):
            dist = pyspike.spike_distance([sti[i], stj[j]])
            if dist < 0:
                continue
            if np.abs(dist) > 100:
                print(dist)
                assert np.abs(dist) < 100
                
            ds.append(dist)
            
    ds = np.array(ds)

    if ds.shape[0] == 0:
        return 0.0
            
    assert ds[np.logical_or(np.isnan(ds), np.isinf(ds))].shape[0] == 0
    return np.nanmean(ds[np.logical_not(np.logical_or(np.isnan(ds), np.isinf(ds)))])

def compute_ISI_on_flat_pair(arg):

    sti, stj = arg
    assert sti.shape[0] == stj.shape[0]
    ds = []
    for i in range(sti.shape[0]):
        for j in range(stj.shape[0]):
            ds.append(pyspike.isi_distance([sti[i], stj[j]]))
            
    ds = np.array(ds)
    assert ds[np.logical_or(np.isnan(ds), np.isinf(ds))].shape[0] == 0
    return np.nanmean(ds[np.logical_not(np.logical_or(np.isnan(ds), np.isinf(ds)))])

def format_to_pyspike(trials, stim_dur):

    st = []
    for t in trials:
        st.append(pyspike.SpikeTrain(t, stim_dur))
    return st
    
def get_fit(func):
    mean,std = norm.fit(func)
    xmin, xmax = plt.xlim()
    if std==0:
        std=0.25
            
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)
    return x, y
     
def time_to_amp(arr):
    # Time axis changed to amplitude modulation
    # Max aplitude is 0.5. 0.125 added because of time_adap stim at the end
    return np.multiply(np.ones_like(arr) * 0.625 / 10, arr, out=np.full_like(arr, np.nan, dtype=np.double), where=arr!=np.nan)
