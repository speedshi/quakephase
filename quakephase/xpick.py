


import numpy as np
from scipy.signal import find_peaks
from seisbench.models.base import WaveformModel
import seisbench.util as sbu
from seisbench.util import Pick as sbPick



def get_picks(prob, paras):
    # get picks from probability curves
    
    phase_tags = ['P', 'S']
    picks = sbu.PickList()
    for itag in phase_tags:
        if (paras['pick']['method'].lower()=='threshold'):
            # use a picking threshold 
            picks += WaveformModel.picks_from_annotations(annotations=prob.select(channel=f"*_{itag}"),
                                                          threshold=paras['pick'][f"{itag}_threshold"],
                                                          phase=itag)
        elif (paras['pick']['method'].lower()=='max'):
            # make a pick anyway, recoginze the maximum probability as the threshold
            assert(prob.select(channel=f"*_{itag}").count() == 1)  # should be only one trace
            iprob = prob.select(channel=f"*_{itag}")[0]  # phase probabilities
            iprob_max = np.abs(iprob.max())  # maximum phase probability, treat it as the picking threshold
            trace_id = f"{iprob.stats.network}.{iprob.stats.station}.{iprob.stats.location}"  # the trace id
            ipick_time = iprob.stats.starttime + iprob.times()[iprob.data.argmax()]  # the picking time of this phase
            ipick = sbPick(trace_id=trace_id, start_time=ipick_time, end_time=ipick_time, 
                           peak_time=ipick_time, peak_value=iprob_max, phase=itag)  # take the time at the maximum probability as the pick time
            picks += [ipick]
        elif (paras['pick']['method'].lower()=='peak'):
            # using find_peaks to pick
            assert(prob.select(channel=f"*_{itag}").count() == 1)  # should be only one trace
            iprob = prob.select(channel=f"*_{itag}")[0]  # phase probabilities
            peaks_indx, _ = find_peaks(x=iprob.data, 
                                       height=paras['pick'][f"{itag}_threshold"], # the required height of peaks
                                       threshold=paras['pick']['nb_threshold'], # the vertical distance to its direct neighboring samples
                                       distance=paras['pick']['distance'], # the required minimal horizontal distance (>= 1) in samples between neighbouring peaks
                                       prominence=paras['pick']['prominence'],
                                       width=paras['pick']['width'], 
                                       wlen=paras['pick']['wlen'],
                                       rel_height=paras['pick']['rel_height'], 
                                       plateau_size=paras['pick']['plateau_size'])
            trace_id = f"{iprob.stats.network}.{iprob.stats.station}.{iprob.stats.location}"
            for kk, kpki in enumerate(peaks_indx):
                if paras['pick'][f"{itag}_threshold"] is None:
                    kk_start_time = iprob.stats.starttime + iprob.times()[kpki]
                    kk_end_time = iprob.stats.starttime + iprob.times()[kpki]
                    kk_peak_time = iprob.stats.starttime + iprob.times()[kpki]
                    kk_peak_value = iprob.data[kpki]
                else:
                    # have input threshold
                    assert(iprob.data[kpki]>=paras['pick'][f"{itag}_threshold"])
                    ### TO DO ...
                    kk_start_time = 
                    kk_end_time = 
                    kk_peak_time = 
                    kk_peak_value = 
                kkpick = sbPick(trace_id=trace_id, start_time=kk_start_time, end_time=kk_end_time,
                                peak_time=kk_peak_time, peak_value=kk_peak_value, phase=itag)
                picks += [kkpick]
        else:
            raise ValueError

    return sbu.PickList(sorted(picks))



