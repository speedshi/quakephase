


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
            kflag = 0  # array index
            Nsamp = np.size(iprob.data)  # total number of sample in probability curve
            for kk, kpki in enumerate(peaks_indx):
                if paras['pick'][f"{itag}_threshold"] is None:
                    kk_start_time = iprob.stats.starttime + iprob.times()[kpki]
                    kk_end_time = iprob.stats.starttime + iprob.times()[kpki]
                    kk_peak_time = iprob.stats.starttime + iprob.times()[kpki]
                    kk_peak_value = iprob.data[kpki]
                else:
                    # have input threshold
                    if (kpki < kflag): continue
                    assert(iprob.data[kpki]>=paras['pick'][f"{itag}_threshold"])

                    fgia = np.zeros(Nsamp,dtype=bool)
                    fgia[kflag:] = True
                    usia = (iprob.data >= paras['pick'][f"{itag}_threshold"]) & (fgia)
                    if sum(usia)==0: break
                    start_idx = np.argmax(usia)
                    assert(iprob.data[start_idx] >= paras['pick'][f"{itag}_threshold"])

                    fgia_e = np.zeros(Nsamp,dtype=bool)
                    fgia_e[start_idx:] = True
                    usia_e = (iprob.data < paras['pick'][f"{itag}_threshold"]) & (fgia_e)
                    if sum(usia_e)==0: 
                        end_idx = Nsamp - 1
                    else:
                        end_idx = np.argmax(usia_e) - 1
                    assert(iprob.data[end_idx] >= paras['pick'][f"{itag}_threshold"])

                    if kpki > end_idx:
                        kflag = end_idx + 1
                        if kflag == Nsamp:
                            break
                        else:
                            continue
                    else:
                        assert(kpki>=start_idx)
                        peak_idx = kpki
                        for mm in range(kk+1, len(peaks_indx)):
                            if peaks_indx[mm] > end_idx:
                                break
                            else:
                                if iprob.data[peaks_indx[mm]] > iprob.data[peak_idx]:
                                    peak_idx = peaks_indx[mm]
                    assert(iprob.data[peak_idx] >= paras['pick'][f"{itag}_threshold"])

                    assert(start_idx <= peak_idx <= end_idx)
                    kk_start_time = iprob.stats.starttime + iprob.times()[start_idx]
                    kk_end_time = iprob.stats.starttime + iprob.times()[end_idx]
                    kk_peak_time = iprob.stats.starttime + iprob.times()[peak_idx]
                    kk_peak_value = iprob.data[peak_idx]
                    kflag = end_idx + 1
                    if kflag == Nsamp: break
                kkpick = sbPick(trace_id=trace_id, start_time=kk_start_time, end_time=kk_end_time,
                                peak_time=kk_peak_time, peak_value=kk_peak_value, phase=itag)
                picks += [kkpick]
        else:
            raise ValueError(f"Invalid input for pick method: {paras['pick']['method']}!")

    return sbu.PickList(sorted(picks))



