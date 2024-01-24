


import numpy as np
from scipy.signal import find_peaks
# from seisbench.models.base import WaveformModel
import seisbench.util as sbu
from seisbench.util import Pick as sbPick
from obspy.signal.trigger import trigger_onset



def picks_from_annotations(annotations, threshold, phase) -> sbu.PickList:
    """
    Converts the annotations streams for a single phase to discrete picks using a classical trigger on/off.
    The lower threshold is set to be the same as the higher threshold. *** Modified by Peidong Shi ***
    Picks are represented by :py:class:`~seisbench.util.annotations.Pick` objects.
    The pick start_time and end_time are set to the trigger on and off times.

    :param annotations: Stream of annotations
    :param threshold: Higher threshold for trigger
    :param phase: Phase to label, only relevant for output phase labelling
    :return: List of picks
    """
    picks = []
    for trace in annotations:
        trace_id = (
            f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}"
        )
        triggers = trigger_onset(trace.data, threshold, threshold)  # note here I modified: lower threshold = higher threshold
        times = trace.times()
        for s0, s1 in triggers:
            t0 = trace.stats.starttime + times[s0]
            t1 = trace.stats.starttime + times[s1]

            peak_value = np.max(trace.data[s0 : s1 + 1])
            s_peak = s0 + np.argmax(trace.data[s0 : s1 + 1])
            t_peak = trace.stats.starttime + times[s_peak]

            pick = sbu.Pick(
                trace_id=trace_id,
                start_time=t0,
                end_time=t1,
                peak_time=t_peak,
                peak_value=peak_value,
                phase=phase,
            )
            picks.append(pick)

    return sbu.PickList(sorted(picks))



def get_picks(prob, paras):
    # get picks from probability curves
    
    phase_tags = ['P', 'S']
    picks = sbu.PickList()
    for itag in phase_tags:
        if (paras['pick']['method'].lower()=='threshold'):
            # use a picking threshold 
            # picks += WaveformModel.picks_from_annotations(annotations=prob.select(channel=f"*_{itag}"),
            #                                               threshold=paras['pick'][f"{itag}_threshold"],
            #                                               phase=itag)
            picks += picks_from_annotations(annotations=prob.select(channel=f"*_{itag}"),
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
            peaks_indx.sort()
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



def picks_clean(picks, phase_min_time):
    '''
    picks: list of SeisBench Pick objects;
    phase_min_time: float in second, only one pick can exist within this time range;
    '''

    picks = sbu.PickList(sorted(picks))  # picks must be ordered according to picking time

    picks_c = sbu.PickList()  # cleaned list
    Npk = len(picks)
    exclude_pkidx_list = []  # picks to exclude
    for ii, ipick  in enumerate(picks):
        if ii in exclude_pkidx_list: continue
        add_this_pick = True
        for jj in range(ii+1, Npk):
            # check all the remaniing picks
            pktime_diff = abs(picks[jj].peak_time - ipick.peak_time)  # absolute picking differential time in second
            if pktime_diff <= phase_min_time:
                # within the preset limit
                # will keep the one with a larger picking probability
                if ipick.peak_value < picks[jj].peak_value:
                    # ipick has a smaller picking probability
                    # discard ipick 
                    add_this_pick = False
                    break
                else:
                    # ipick have a larger picking probability
                    # the tested picks[jj] will be discard
                    # continoue checking
                    exclude_pkidx_list.append(jj)
            else:
                # already outside the limit, since picks list are oderded, no need to checking the following
                break

        if add_this_pick: picks_c += [ipick]  # add the current ipick if it pass the criterion

    return sbu.PickList(sorted(picks_c))



