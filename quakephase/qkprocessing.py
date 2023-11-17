

import bottleneck as bn
from obspy.core import Stream
import numpy as np
from sklearn.decomposition import PCA



def sbresample(stream, sampling_rate):
    """
    Perform inplace resampling of stream to a given sampling rate.
    The same as used in SeisBench.

    :param stream: Input stream
    :type stream: obspy.core.Stream
    :param sampling_rate: Sampling rate (sps) to resample to
    :type sampling_rate: float
    """
    del_list = []
    for i, trace in enumerate(stream):
        if trace.stats.sampling_rate == sampling_rate:
            continue
        if trace.stats.sampling_rate % sampling_rate == 0:
            trace.filter("lowpass", freq=sampling_rate * 0.5, zerophase=True)
            trace.decimate(
                int(trace.stats.sampling_rate / sampling_rate), no_filter=True
            )
        else:
            trace.resample(sampling_rate, no_filter=True)

    for i in del_list:
        del stream[i]
    
    return


def stfilter(stream, fband):
    # filter seismic stream
    # operation is performed in place
    # so no return parameters
    
    stream.detrend('demean')
    stream.detrend('simple')
    # stream.detrend('linear')
    stream.filter('bandpass', freqmin=fband[0], freqmax=fband[1], corners=2, zerophase=True)
    #stream.taper(max_percentage=0.01, type='cosine', max_length=0.1)  # to avoid anormaly at boundary
    
    return


def prob_ensemble(probs_all, method="max", sampling_rate=None):
    '''
    Aggregate results from different predictions.
    '''

    munify = 'prob'

    prob_starttime = []
    prob_endtime = []
    prob_sampling_rate = []
    for iprob in probs_all:
        iprob.merge(method=1, interpolation_samples=-1, fill_value=None)
        for itr in iprob:
            prob_starttime.append(itr.stats.starttime)
            prob_endtime.append(itr.stats.endtime)
            prob_sampling_rate.append(itr.stats.sampling_rate)

    prob_starttime = max(prob_starttime)  # the latest starttime
    prob_endtime = min(prob_endtime)  # the earliest endtime
    if sampling_rate is None:
        prob_sampling_rate = max(prob_sampling_rate)  # use the maximum sampling_rate
    elif isinstance(sampling_rate, (int,float)):
        prob_sampling_rate = sampling_rate  # use input sampling rate
    else:
        raise ValueError(f"Invalid input of sampling rate: {sampling_rate}!")

    for iprob in probs_all:
        # resample prob to the same frequency sampling_rate
        sbresample(stream=iprob, sampling_rate=prob_sampling_rate)

        # trim prob to the same time range -> prob_starttime - prob_endtime
        for jj, jjtr in enumerate(iprob):
            tr_temp = jjtr.copy()
            tr_temp.trim(starttime=prob_starttime, endtime=prob_endtime, pad=True,
                         nearest_sample=False, fill_value=None)
            if (tr_temp.stats.starttime == prob_starttime):
                iprob[jj] = tr_temp.copy()
                Nsamp = len(iprob[jj].data)  # total number of data samples
    
    prob = Stream()  # empty stream
    pdata = {}  # probability datasets
    for iprob in probs_all:
        for itr in iprob:
            if (itr.stats.starttime != prob_starttime):
                itr.interpolate(sampling_rate=prob_sampling_rate,
                                method="weighted_average_slopes",
                                starttime=prob_starttime, npts=Nsamp)
            assert(abs(itr.stats.sampling_rate-prob_sampling_rate)<1E-8)
            assert(itr.stats.starttime==prob_starttime)
            assert(itr.data.shape==(Nsamp,))
            itag = itr.stats.channel.split('_')[-1]  # phase or classify tage
            if itag not in pdata:
                pdata[itag] = itr.data.reshape(-1,1)
                itr_m = itr.copy()
                itr_m.stats.channel = f"{munify}_{itag}"  # renew channel name
                prob.append(itr_m)
            else:
                pdata[itag] = np.hstack((pdata[itag], itr.data.reshape(-1,1)))  # data shape: n_samples * n_probs

    if method.lower() == "max":
        for ikey in pdata.keys():
            xprob = prob.select(channel=f"*_{ikey}")
            assert(xprob.count()==1)  # only one
            assert(xprob[0].stats.starttime==prob_starttime)
            assert(abs(xprob[0].stats.sampling_rate-prob_sampling_rate)<1E-8)
            assert(xprob[0].data.shape==(Nsamp,))
            xprob[0].data = bn.nanmax(pdata[ikey], axis=-1)  # <._.>
            assert(xprob[0].data.shape==(Nsamp,))
    elif method.lower() == "min":
        for ikey in pdata.keys():
            xprob = prob.select(channel=f"*_{ikey}")
            assert(xprob.count()==1)  # only one
            assert(xprob[0].stats.starttime==prob_starttime)
            assert(abs(xprob[0].stats.sampling_rate-prob_sampling_rate)<1E-8)
            assert(xprob[0].data.shape==(Nsamp,))
            xprob[0].data = bn.nanmin(pdata[ikey], axis=-1)  # <._.>
            assert(xprob[0].data.shape==(Nsamp,))
    elif method.lower() == "mean":
        for ikey in pdata.keys():
            xprob = prob.select(channel=f"*_{ikey}")
            assert(xprob.count()==1)  # only one
            assert(xprob[0].stats.starttime==prob_starttime)
            assert(abs(xprob[0].stats.sampling_rate-prob_sampling_rate)<1E-8)
            assert(xprob[0].data.shape==(Nsamp,))
            xprob[0].data = bn.nanmean(pdata[ikey], axis=-1)  # <._.>
            assert(xprob[0].data.shape==(Nsamp,))
    elif method.lower() == "median":
        for ikey in pdata.keys():
            xprob = prob.select(channel=f"*_{ikey}")
            assert(xprob.count()==1)  # only one
            assert(xprob[0].stats.starttime==prob_starttime)
            assert(abs(xprob[0].stats.sampling_rate-prob_sampling_rate)<1E-8)
            assert(xprob[0].data.shape==(Nsamp,))
            xprob[0].data = bn.nanmedian(pdata[ikey], axis=-1)  # <._.>
            assert(xprob[0].data.shape==(Nsamp,))
    elif (method.lower() == "prod") or (method.lower() == "multiply"):
        for ikey in pdata.keys():
            xprob = prob.select(channel=f"*_{ikey}")
            assert(xprob.count()==1)  # only one
            assert(xprob[0].stats.starttime==prob_starttime)
            assert(abs(xprob[0].stats.sampling_rate-prob_sampling_rate)<1E-8)
            assert(xprob[0].data.shape==(Nsamp,))
            xprob[0].data = np.nanprod(pdata[ikey], axis=-1)  # <._.>
            assert(xprob[0].data.shape==(Nsamp,))
    elif (method.lower() == "semblance"):
       wdp = 20
       bup = 2
       for ikey in pdata.keys():
            xprob = prob.select(channel=f"*_{ikey}")
            assert(xprob.count()==1)  # only one
            assert(xprob[0].stats.starttime==prob_starttime)
            assert(abs(xprob[0].stats.sampling_rate-prob_sampling_rate)<1E-8)
            assert(xprob[0].data.shape==(Nsamp,))
            nt, npb = pdata[ikey].shape
            weit = bn.nanmax(pdata[ikey], axis=-1)
            square_of_sums = bn.nansum(pdata[ikey], axis=-1)**2
            sum_of_squares = bn.nansum(pdata[ikey]**2, axis=-1)
            xprob[0].data[:] = 0
            for jj in range(wdp,nt-wdp-1):
                xprob[0].data[jj] = square_of_sums[jj-wdp:jj+wdp+1].sum() / sum_of_squares[jj-wdp:jj+wdp+1].sum() / npb  # <._.>
            xprob[0].data = (xprob[0].data**bup) * weit
            assert(xprob[0].data.shape==(Nsamp,))
    elif (method.lower() == "pca"):
        pca = PCA(n_components=1)
        for ikey in pdata.keys():
            xprob = prob.select(channel=f"*_{ikey}")
            assert(xprob.count()==1)  # only one
            assert(xprob[0].stats.starttime==prob_starttime)
            assert(abs(xprob[0].stats.sampling_rate-prob_sampling_rate)<1E-8)
            assert(xprob[0].data.shape==(Nsamp,))
            pbmin = bn.nanmin(pdata[ikey], axis=None)  # minimal value of original data
            pbmax = bn.nanmax(pdata[ikey], axis=None)  # maximum value of original data
            xprob[0].data = pca.fit_transform(pdata[ikey])[:,0]
            xmin = bn.nanmin(xprob[0].data, axis=None)  # minimal value of transferred data
            xmax = bn.nanmax(xprob[0].data, axis=None)  # maximal value of transferred data
            xprob[0].data = pbmin + (xprob[0].data - xmin) * (pbmax - pbmin) / (xmax - xmin)  # scale data back to original range
            assert(xprob[0].data.shape==(Nsamp,))
    else:
        ### TO DO: add more emsemble methods: Bayesian, Kalman Filtering, etc
        raise ValueError(f'Invalid input for ensemble method: {method}!')

    # recompile noise probability: Noise_prob = 1 - P_prob - S_prob
    if 'N' in pdata:
        Nprob = prob.select(channel='*_N')
        assert(Nprob.count()==1)  # only one
        Pprob = prob.select(channel='*_P')
        assert(Pprob.count()==1)  # only one
        Sprob = prob.select(channel='*_S')
        assert(Sprob.count()==1)  # only one
        Nprob[0].data = 1 - (Pprob[0].data + Sprob[0].data)

    return prob



def check_compile_stream(istream):

    # check and compile istream when necessary
    istream.merge(method=1, fill_value=0)  # merge data with the same channel
    comps = [itr.stats.channel[-1].upper() for itr in istream]  # avaliable components in the input data

    # require certain components
    accept_comps = ['Z', 'N', 'E', '1', '2', '3']
    while (not set(comps).issubset(set(accept_comps))):
        # rename the components that are not in accept_comps
        for kk in range(istream.count()):
            if istream[kk].stats.channel[-1].upper() not in accept_comps:
                if 'Z' not in comps:
                    istream[kk].stats.channel = istream[kk].stats.channel[:-1]+'Z'
                    comps = [itr.stats.channel[-1].upper() for itr in istream]  # renew comps
                elif ('1' in comps) or ('2' in comps) or ('3' in comps):
                    for icp in ['1', '2', '3']:
                        if icp not in comps:
                            istream[kk].stats.channel = istream[kk].stats.channel[:-1]+icp
                            comps = [itr.stats.channel[-1].upper() for itr in istream]  # renew comps
                            break
                else:
                    for icp in ['N', 'E']:
                        if icp not in comps:
                            istream[kk].stats.channel = istream[kk].stats.channel[:-1]+icp
                            comps = [itr.stats.channel[-1].upper() for itr in istream]  # renew comps
                            break

    # require 3-components inputs
    while (istream.count()<3):
        # append a new trace to make 3-components
        if ('Z' not in comps):
            itrace = istream[0].copy()
            itrace.stats.channel = istream[0].stats.channel[:-1]+'Z'  # append a 'Z' trace
            itrace.data[:] = 0
            istream.append(itrace.copy())
            comps = [itr.stats.channel[-1].upper() for itr in istream]  # renew comps
        elif ('1' in comps) or ('2' in comps) or ('3' in comps):
            for icp in ['1', '2', '3']:
                if icp not in comps:
                    itrace = istream[0].copy()
                    itrace.stats.channel = istream[0].stats.channel[:-1]+icp
                    itrace.data[:] = 0
                    istream.append(itrace.copy())
                    comps = [itr.stats.channel[-1].upper() for itr in istream]  # renew comps
                    break
        else:
            for icp in ['N', 'E']:
                if icp not in comps:
                    itrace = istream[0].copy()
                    itrace.stats.channel = istream[0].stats.channel[:-1]+icp
                    itrace.data[:] = 0
                    istream.append(itrace.copy())
                    comps = [itr.stats.channel[-1].upper() for itr in istream]
                    break

    return istream


