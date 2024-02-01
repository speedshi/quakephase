

import obspy
from obspy import UTCDateTime
import numpy as np
import math



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
    if (fband[0] is None) and (isinstance(fband[1], (int,float))):
        stream.filter('lowpass', freq=fband[1], corners=2, zerophase=True)
    elif (fband[1] is None) and (isinstance(fband[0], (int,float))):
        stream.filter('highpass', freq=fband[0], corners=2, zerophase=True)
    elif (isinstance(fband[0], (int,float))) and (isinstance(fband[1], (int,float))):
        stream.filter('bandpass', freqmin=fband[0], freqmax=fband[1], corners=2, zerophase=True)
    else:
        raise ValueError(f"Invalid input for fband: {fband}!")
    #stream.taper(max_percentage=0.01, type='cosine', max_length=0.1)  # to avoid anormaly at boundary
    
    return


def check_compile_stream(istream):

    # check and compile istream when necessary
    istream.merge(method=1, fill_value=0)  # merge data with the same channel
    comps = [itr.stats.channel[-1].upper() for itr in istream]  # avaliable components in the input data

    # check input components, we only accept certain components, rename input components if necessary
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

    # # require 3-components inputs
    # while (istream.count()<3):
    #     # append a new trace to make 3-components
    #     if ('Z' not in comps):
    #         itrace = istream[0].copy()
    #         itrace.stats.channel = istream[0].stats.channel[:-1]+'Z'  # append a 'Z' trace
    #         itrace.data[:] = 0
    #         istream.append(itrace.copy())
    #         comps = [itr.stats.channel[-1].upper() for itr in istream]  # renew comps
    #     elif ('1' in comps) or ('2' in comps) or ('3' in comps):
    #         for icp in ['1', '2', '3']:
    #             if icp not in comps:
    #                 itrace = istream[0].copy()
    #                 itrace.stats.channel = istream[0].stats.channel[:-1]+icp
    #                 itrace.data[:] = 0
    #                 istream.append(itrace.copy())
    #                 comps = [itr.stats.channel[-1].upper() for itr in istream]  # renew comps
    #                 break
    #     else:
    #         for icp in ['N', 'E']:
    #             if icp not in comps:
    #                 itrace = istream[0].copy()
    #                 itrace.stats.channel = istream[0].stats.channel[:-1]+icp
    #                 itrace.data[:] = 0
    #                 istream.append(itrace.copy())
    #                 comps = [itr.stats.channel[-1].upper() for itr in istream]
    #                 break
    # assert(istream.count()==3)

    return istream


def array2stream(data, paras):
    # convert numpy arrays to obspy stream
    # data should be a 2D array, with shape (Nsamples, Ntraces)
    # Ntraces = Ncomponents * Nstations
    # paras is a dict, with keys: component_input
    # paras['component_input']: the input component codes, e.g., 'Z12' or 'ZNE' or 'Z'

    # set default required paramters, these does not affect model performance
    starttime = UTCDateTime(0)  # fixed, do not change!
    sampling_rate = 100  # Hz, keep the same as the default value for SeisBench ML models
    network_code = 'XX'
    station_code_prefix = 'X'
    location_code = '00'
    instrument_code = 'HH'

    component_codes = paras['component_input']
    nc = len(component_codes)  # number of input components

    stream = obspy.Stream()
    if data.ndim == 1:
        data = data.reshape((-1, 1))
    elif data.ndim == 2:
        pass
    elif data.ndim == 3:
        data = data.reshape((data.shape[0], -1))
    else:
        raise ValueError(f"Invalid input data shape: {data.shape}! Must be 1D, 2D or 3D.")
    (Nsamples, Ntraces) = data.shape
    if Ntraces%nc != 0:
        raise ValueError(f"Invalid input data shape: {data.shape}! Must be mutiples of input components: {nc}.")
    else:
        Nstations = Ntraces//nc
        print(f"Total number of input components: {nc}, [{component_codes}].")
        print(f"Total number of input traces: {Ntraces}.")
        print(f"Total number of input samples: {Nsamples}.")
        print(f"Total number of input stations: {Nstations}.")

    for ii in range(Ntraces):
        istation = ii//nc  # the station index
        icomp = ii%nc  # the component index
        station_code = f"{station_code_prefix}{istation}"
        channel_code = f"{instrument_code}{component_codes[icomp]}"
        print(f"Format input data for station: {station_code}, channel: {channel_code}.")
        itrace = obspy.Trace(data=data[:, ii],
                             header={'starttime': starttime,
                                     'sampling_rate': sampling_rate,
                                     'network': network_code,
                                     'station': station_code,
                                     'location': location_code,
                                     'channel': channel_code,
                                     })
        stream.append(itrace)

    return stream


def expend_trace(trace, window_in_second, method):

    # expend trace by padding at the begining and end
    Nsamp = math.ceil(trace.stats.sampling_rate * window_in_second)  # total number of samples required
    Ninputs = trace.stats.npts  # total number of input samples
    assert(Nsamp > Ninputs)
    trace_starttime = trace.stats.starttime
    npad_half = math.ceil((Nsamp - Ninputs)*0.5)  # the number of samples to be padded at the begining and end
    trace.data = np.pad(array=trace.data, pad_width=(npad_half,npad_half), mode=method)
    trace.stats.starttime = trace_starttime - npad_half*trace.stats.delta  # shift the start time to compensate the padding
    assert(trace.stats.npts >= Nsamp)
    return trace



