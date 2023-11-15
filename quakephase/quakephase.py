

from .load_MLmodel import load_MLmodel
from .qkprocessing import stfilter, prob_ensemble, check_compile_stream
from .pfinput import load_check_input
from .xpick import get_picks
import obspy
import seisbench.util as sbu
import pandas as pd



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
    
    # check
    assert(len(phasemodels)==Nmlmd*Nresc)

    # load seismic data
    if isinstance(data, (obspy.Stream)):
        stream = data
    elif isinstance(data, (str)):
        stream = obspy.read(data)
    elif isinstance(data, (list)):
        stream = obspy.Stream()
        for idata in data:
            assert(isinstance(idata, (str)))
            stream += obspy.read(idata)

    # apply model to data streams, loop over each station
    station_list = []
    for itr in stream:
        station_list.append(itr.stats.station)
    station_list = list(set(station_list))  # remove duplicates

    output = {}
    if (paras['output'].lower() == 'prob') or (paras['output'].lower() == 'all'):
        output['prob'] = obspy.Stream()
    elif (paras['output'].lower() == 'pick') or (paras['output'].lower() == 'all'):
        output['pick'] = []
    else:
        raise ValueError(f"Unrecognized output type {paras['output']}!")

    for istation in station_list:
        istream = stream.select(station=istation).copy()  # data for a single station
        istream = check_compile_stream(istream)  # check and compile stream, need 3-component data
 
        assert(istream.count()==3)
        ioutput = apply_per_station(istream, phasemodels, paras)

        # append results to output
        if (paras['output'].lower() == 'prob') or (paras['output'].lower() == 'all'):
            output['prob'] += ioutput['prob']
        elif (paras['output'].lower() == 'pick') or (paras['output'].lower() == 'all'):
            output['pick'] += ioutput['pick']
        else:
            raise ValueError(f"Unrecognized output type {paras['output']}!")
        
        # delete istream to save memory
        del istream

    if (paras['output'].lower() == 'pick') or (paras['output'].lower() == 'all'):
        output['pick'] = sbu.PickList(sorted(output['pick']))  # sort picks
        
        # format pick output to specified format
        if paras['pick']['format'].lower() == 'list':
            output['pick'] = [ipick.__dict__ for ipick in output['pick']]  # convert to dict
        elif paras['pick']['format'].lower() == 'dict':
            # convert list of dict to dict of list
            output['pick'] = [ipick.__dict__ for ipick in output['pick']]  # convert to dict
            output['pick'] = {k: [d[k] for d in output['pick']] for k in output['pick'][0]}
        elif paras['pick']['format'].lower() == 'dataframe':
            # convert list of dict to dataframe
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
        for ifreq in paras['frequency']:
            stream_ft = istream.copy()
            if (isinstance(ifreq, (list))):
                # filter data in specified frequency range, bandpass filter
                stfilter(stream_ft, fband=ifreq)                

            # obtain phase probability for each model and frequency
            probs_all.append(kmodel.annotate(stream=stream_ft))
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
        # pick_check = phasemodels[0].classify(stream, P_threshold=paras['pick']['P_threshold'],
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



