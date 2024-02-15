

from .load_MLmodel import load_MLmodel
from .xprob import prob_ensemble
from .streamprocess import stfilter, check_compile_stream, array2stream, expend_trace
from .pfinput import load_check_input
from .xpick import get_picks
import obspy
import seisbench.util as sbu
import pandas as pd
import numpy as np
from obspy import UTCDateTime



def apply(data, file_para='parameters.yaml'):
    '''
    INPUT:
        data: obspy stream object or str or list of str;
              if data is str or list of str, then it should be the path to the seismic data file(s)
              which obspy can read;
        file_para: str, path to the paramter YAML file for quakephase;

    OUTPUT:
        output: dict, contains the following keys:
            'prob': obspy stream object, phase probability for each station;
            'pick': list of xpick object, phase picks for each station.
    '''

    # load and check input parameters
    paras = load_check_input(file_para=file_para)

    Nmlmd = len(paras['MLmodel'])  # total number of used ML models
    Nresc = len(paras['rescaling'])  # total number of rescaling rates

    # load ML models
    phasemodels = []
    for imodel in paras['MLmodel']:  # loop over each model id
        for irescaling in paras['rescaling']:  # loop over each rescaling_rate
            phasemodels.append(load_MLmodel(model_id=imodel, rescaling_rate=irescaling, 
                                            overlap_ratio=paras['overlap_ratio'], blinding=None))  # blinding=(0, 0)

    # check number of loaded models
    assert(len(phasemodels)==Nmlmd*Nresc)

    # format output
    output = {}
    if (paras['output'].lower() == 'prob') or (paras['output'].lower() == 'all'):
        output['prob'] = obspy.Stream()
    if (paras['output'].lower() == 'pick') or (paras['output'].lower() == 'all'):
        output['pick'] = []

    # load seismic data
    if isinstance(data, (obspy.Stream)):
        stream = data
    elif isinstance(data, (obspy.Trace)):
        stream = obspy.Stream(traces=[data])
    elif isinstance(data, (str)):
        stream = obspy.read(data)
    elif isinstance(data, (list)) and all(isinstance(item, str) for item in data):
        # input are paths to seismic data files, a list of str
        stream = obspy.Stream()
        for idata in data:
            assert(isinstance(idata, (str)))
            stream += obspy.read(idata)
    elif isinstance(data, (np.ndarray)):
        # input are numpy arrays
        # convert numpy arrays to obspy stream
        # data should be a 2D array, with shape (Nsamples, Ntraces)
        stream = array2stream(data=data, paras=paras['data'])
        paras['prob_sampling_rate'] = 100  # Hz, reset probability data sampling rate to the default value
    else:
        raise ValueError(f"Unknown data type: {type(data)}")    

    # get station list in the stream data
    station_list = []
    for itr in stream:
        station_list.append(itr.stats.station)
    station_list = list(set(station_list))  # remove duplicates

    # apply model to data streams, loop over each station
    for istation in station_list:
        istream = stream.select(station=istation).copy()  # data for a single station
        istream = check_compile_stream(istream)  # check and compile stream
        ioutput = apply_per_station(istream, phasemodels, paras)

        # append results to output
        if (paras['output'].lower() == 'prob') or (paras['output'].lower() == 'all'):
            output['prob'] += ioutput['prob']
        if (paras['output'].lower() == 'pick') or (paras['output'].lower() == 'all'):
            output['pick'] += ioutput['pick']
        
        # delete istream to save memory
        del istream

    if (paras['output'].lower() == 'pick') or (paras['output'].lower() == 'all'):
        output['pick'] = sbu.PickList(sorted(output['pick']))  # sort picks
        
        # need to reformate picks if input data format is numpy.ndarray
        # convert picking results of UCTDateTime to samples
        # the reference time is always UTCDateTime(0)
        # default sampling rate is 100 Hz
        if isinstance(data, (np.ndarray)):
            for jjpick in output['pick']:
                # loop over each pick
                for attr, avalue in vars(jjpick).items():
                    # loop over each attribute of the pick
                    if isinstance(avalue, UTCDateTime):
                        # a picking time related attribute
                        new_avalue = (avalue - UTCDateTime(0)) * 100 + 1 # convert to samples
                        setattr(jjpick, attr, new_avalue)

        # format pick output to specified format
        if paras['pick']['format'] is None:
            pass
        elif paras['pick']['format'].lower() == 'list':
            # output is list of pick_dict
            output['pick'] = [ipick.__dict__ for ipick in output['pick']]  # convert to dict
        elif paras['pick']['format'].lower() == 'dict':
            # output is dict of pick_list
            output['pick'] = [ipick.__dict__ for ipick in output['pick']]  # convert to dict
            output['pick'] = {k: [d[k] for d in output['pick']] for k in output['pick'][0]}
        elif paras['pick']['format'].lower() == 'dataframe':
            # output is pick dataframe
            output['pick'] = [ipick.__dict__ for ipick in output['pick']]  # convert to dict
            output['pick'] = pd.DataFrame(output['pick'])

    return output



def apply_per_station(istream, phasemodels, paras):
    '''
    Apply model to data streams.
    INPUT:
        istream: obspy stream object, should be a single station data;
        phasemodels: list of phase ML model objects;
        paras: dict, contains the following keys:
            'frequency': list of frequency ranges, e.g., [None, [1, 10], [10, 20], [20, 50]];
            'prob_sampling_rate': None or float, sampling rate for the output probability stream;
            'ensemble': str, method for ensemble, 'pca, 'max', 'semblance', ''mean' or 'median';
            'output': str, output type, 'prob', 'pick' or 'all'.
    '''

    probs_all = []
    for kmodel in phasemodels:
        # loop over each model
        pdtw = kmodel.in_samples / float(kmodel.sampling_rate)  # prediction window length of the model, in seconds

        for ifreq in paras['frequency']:
            # loop over each frequency range
            stream_ft = istream.copy()

            # auto expend data if required
            if ('auto_expend' in paras['data']):
                # auto expend data to the required length if input is not enough
                pdtw_used = pdtw * paras['data']['auto_expend']['window_ratio']
                trace_expended = False
                itrace_starttime_min = stream_ft[0].stats.starttime
                itrace_endtime_max = stream_ft[0].stats.endtime
                for jjtr in range(stream_ft.count()):
                    if stream_ft[jjtr].stats.starttime < itrace_starttime_min:
                        itrace_starttime_min = stream_ft[jjtr].stats.starttime
                    if stream_ft[jjtr].stats.endtime > itrace_endtime_max:
                        itrace_endtime_max = stream_ft[jjtr].stats.endtime
                    if (stream_ft[jjtr].stats.endtime - stream_ft[jjtr].stats.starttime) < pdtw_used:
                        # need to expend data
                        stream_ft[jjtr] = expend_trace(trace=stream_ft[jjtr], window_in_second=pdtw_used, method=paras['data']['auto_expend']['method'])
                        trace_expended = True

            # filter data
            if (isinstance(ifreq, (list))):
                # filter data in specified frequency range
                stfilter(stream_ft, fband=ifreq)

            # obtain phase probability for each model and frequency
            kprob = kmodel.annotate(stream=stream_ft)
            if ('auto_expend' in paras['data']) and (trace_expended):
                # need to trim probability data to the original length
                kprob.trim(starttime=itrace_starttime_min, endtime=itrace_endtime_max, nearest_sample=True)
            probs_all.append(kprob)
            del stream_ft
    
    Nfreq = len(paras['frequency'])  # total number of frequency ranges
    assert(len(probs_all)==len(phasemodels)*Nfreq)

    if len(probs_all) == 1:
        prob = probs_all[0]
    else:
        # remove potential empty prob_streams
        for iprob in probs_all:
            for itrace in iprob:
                if (itrace.count()==0): iprob.remove(itrace)
        probs_all = [iprob for iprob in probs_all if iprob.count()>0]

        if len(probs_all) > 1:            
            # aggregate results from different models/predictions
            prob = prob_ensemble(probs_all=probs_all, method=paras['ensemble'], sampling_rate=paras['prob_sampling_rate'])
        else:
            prob = probs_all[0]

    ioutput = {}
    if (paras['output'].lower() == 'prob') or (paras['output'].lower() == 'all'):
        ioutput['prob'] = prob

    if (paras['output'].lower() == 'pick') or (paras['output'].lower() == 'all'):
        # get picks   
        ioutput['pick'] = get_picks(prob=prob, paras=paras)

        # # check
        # pick_check = phasemodels[0].classify(istream, P_threshold=paras['pick']['P_threshold'],
        #                                      S_threshold=paras['pick']['S_threshold'])
        # assert(len(ioutput['pick'])==len(pick_check.picks))
        # for hh, hhpick in enumerate(ioutput['pick']):
        #     print(pick_check.picks[hh], hhpick)
        #     print()
        #     print(pick_check.picks[hh].__dict__, hhpick.__dict__)
        #     print()
        #     assert(pick_check.picks[hh].__dict__ == hhpick.__dict__)
        #     print('they are the same')

    return ioutput



