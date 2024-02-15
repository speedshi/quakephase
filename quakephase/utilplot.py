

'''
Some plotting utilities for QuakePhase.
'''



from matplotlib import pyplot as plt
import numpy as np
from obspy import UTCDateTime
from datetime import datetime
import seisbench.util as sbu
import pandas as pd



def waveform_pick_1sta(stream, pick, prob=None, pick_threshold=None, time_range=None, prob_range=[0, 1], fband=None, figsize=(12,6), fname=None):

    pick_colors = {}
    pick_colors['P'] = 'r'  # color for P phase
    pick_colors['S'] = 'b'  # color for S phase
    phase_tage = ['P', 'S']  # plotted phases

    ntrace = stream.count()
    if prob is None:
        nrows = ntrace
    else:
        nrows = ntrace + 1

    if time_range is not None:
        if isinstance(time_range[0], (datetime,)):
            plot_time_range = [time_range[0], time_range[1]]
        elif isinstance(time_range[0], (UTCDateTime,)):
            plot_time_range = [time_range[0].datetime, time_range[1].datetime]
        elif isinstance(time_range[0], (float, int)):
            pick_all = []
            for ipick in pick:
                if isinstance(ipick, (sbu.PickList,)):
                    ipick_peak_time = ipick.peak_time
                elif isinstance(ipick, (list,)):
                    ipick_peak_time = ipick['peak_time']
                elif isinstance(ipick, (pd.DataFrame,)):
                    ipick_peak_time = ipick.iloc[ipk]
                else:
                    raise ValueError("pick must be a sbu.PickList, list or pd.DataFrame.")
                pick_all.append(ipick_peak_time)
            pick_min = np.min(pick_all).datetime
            pick_max = np.max(pick_all).datetime    
            plot_time_range = [pick_min + time_range[0], pick_max + time_range[1]]
        else:
            raise ValueError("time_range must be a list of UTCDateTime or float or int.")
    else:
        plot_time_range = None

    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, gridspec_kw={'hspace': 0}, sharex=True)

    # plot waveform and pick
    for ii, trace in enumerate(stream):
        assert(trace.stats.station == stream[0].stats.station)  # input must from the same station
        itrace = trace.copy()

        if fband is not None:
            itrace.detrend('simple') 
            itrace.filter('bandpass', freqmin=fband[0], freqmax=fband[1])

        if plot_time_range is not None:
            itrace.trim(starttime=UTCDateTime(plot_time_range[0]), endtime=UTCDateTime(plot_time_range[1]))

        tt = np.array([(itrace.stats.starttime + dtt).datetime for dtt in itrace.times()])  # time-axis, datetime
        axs[ii].plot(tt, itrace.data, linewidth=1.2, color='k', label=f"{trace.stats.channel}")

        ml_phases = []
        for ipk in range(len(pick)):
            if isinstance(pick, (sbu.PickList,)):
                ipick_time = pick[ipk].peak_time
                ipick_phase = pick[ipk].phase
            elif isinstance(pick, (list,)):
                ipick_time = pick[ipk]['peak_time']
                ipick_phase = pick[ipk]['phase']
            elif isinstance(pick, (pd.DataFrame,)):
                ipick_time = pick['peak_time'].iloc[ipk]
                ipick_phase = pick['phase'].iloc[ipk]
            else:
                raise ValueError("pick must be a sbu.PickList, list or pd.DataFrame.") 

            if ipick_phase not in ml_phases:
                ml_phases.append(ipick_phase)
                axs[ii].axvline(x=ipick_time, color=pick_colors[f"{ipick_phase}"],
                                label=f"{ipick_phase}", linewidth=0.6, linestyle='--')
            else:
                axs[ii].axvline(x=ipick_time, color=pick_colors[f"{ipick_phase}"],
                                linewidth=0.6, linestyle='--')

        axs[ii].legend(loc='upper right')
        axs[ii].set_ylabel('Amplitude', fontsize=12)
        axs[ii].set_xlim(plot_time_range)
        if (prob is None) and (ii == ntrace-1):
            axs[ii].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False)
        else:
            axs[ii].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)

    # plot probabilities and picks
    if prob is not None:
        for itag in phase_tage:
            iprob = prob.select(channel=f"*{itag}")[0].copy()  # probability of a certain phase

            if plot_time_range is not None: 
                iprob.trim(starttime=UTCDateTime(plot_time_range[0]), endtime=UTCDateTime(plot_time_range[1]))

            tt_prob = np.array([(iprob.stats.starttime + dtt).datetime for dtt in iprob.times()])
            axs[nrows-1].plot(tt_prob, iprob.data, linewidth=1.0, color=pick_colors[itag],
                              alpha=0.8, label=f"{iprob.stats.channel}")
            
        for ipk in range(len(pick)):
            if isinstance(pick, (sbu.PickList,)):
                ipick_time = pick[ipk].peak_time
                ipick_phase = pick[ipk].phase
            elif isinstance(pick, (list,)):
                ipick_time = pick[ipk]['peak_time']
                ipick_phase = pick[ipk]['phase']
            elif isinstance(pick, (pd.DataFrame,)):
                ipick_time = pick['peak_time'].iloc[ipk]
                ipick_phase = pick['phase'].iloc[ipk]
            else:
                raise ValueError("pick must be a sbu.PickList, list or pd.DataFrame.") 
            
            axs[nrows-1].axvline(x=ipick_time, color=pick_colors[f"{ipick_phase}"],
                                 alpha=0.8, linewidth=0.5, linestyle='--')

        if pick_threshold is not None: 
            axs[nrows-1].axhline(y=pick_threshold, color='k', linestyle='--', linewidth=0.5, label='threshold')
        
        axs[nrows-1].legend()
        axs[nrows-1].set_ylabel('Probability', fontsize=12)
        axs[nrows-1].set_xlim(plot_time_range)
        axs[nrows-1].set_ylim(prob_range)
        axs[nrows-1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False)

    # save or show figure
    if fname is None:
        plt.show()
    elif isinstance(fname, (str,)):
        fig.savefig(fname=fname, dpi=600, bbox_inches='tight')
        plt.cla()
        fig.clear()
        plt.close(fig)
    else:
        raise ValueError("fname must be a string or None.")

    return





