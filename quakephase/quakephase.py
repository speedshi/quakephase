

import yaml
from .load_MLmodel import load_MLmodel
from .qkprocessing import stfilter, prob_ensemble
import seisbench.util as sbu
from seisbench.models.base import WaveformModel
from seisbench.util import Pick as sbPick
import numpy as np



def load_check_input(file_para):

    # load paramters
    with open(file_para, 'r') as file:
        paras = yaml.safe_load(file)

    assert(isinstance(paras['MLmodel'], (list,)))
    assert(isinstance(paras['rescaling'], (list,)))
    assert(isinstance(paras['frequency'], (list,)))

    # check 'frequency' setting
    for ifreq in paras['frequency']:
        if isinstance(ifreq, (str,)): 
            assert(ifreq.lower()=='none')
        elif isinstance(ifreq, (list,)):
            assert(len(ifreq)==2)
        else:
            raise ValueError(f"Invalid input for frequency paramter: {ifreq}!")

    # check 'output' setting
    if paras['output'].lower() not in ['prob', 'pick', 'all']:
        raise ValueError(f"Unrecognized output type {paras['output']}!")
    
    # check 'pick' setting
    if 'pick' in paras:
        for itag in ['P_threshold', 'S_threshold']:
            if isinstance(paras['pick'][itag],(int,float)):
                assert(0<=paras['pick'][itag]<=1)
            elif isinstance(paras['pick'][itag],(str)):
                assert(paras['pick'][itag].lower()=='max')
            else:
                raise ValueError(f"Invalid input for pick_{itag}:{paras['pick'][itag]}!")

    # check 'prob_sampling_rate' setting
    if isinstance(paras['prob_sampling_rate'], str):
        if paras['prob_sampling_rate'].lower() == "none":
            paras['prob_sampling_rate'] = None
        else:
            raise ValueError(f"Invalid input for prob_sampling_rate {paras['prob_sampling_rate']}!")
    elif isinstance(paras['prob_sampling_rate'], (int,float)):
        pass
    else:
        raise ValueError(f"Invalid input for prob_sampling_rate {paras['prob_sampling_rate']}!")

    return paras


def qkphase(stream, file_para='parameters.yaml'):
    '''
    INPUT:
        stream: three-component obspy stream object;
        file_para: str, path to the paramter YAML file;

    OUTPUT:

    '''

    # load and check input parameters
    paras = load_check_input(file_para=file_para)

    Nmlmd = len(paras['MLmodel'])  # total number of used ML models
    Nresc = len(paras['rescaling'])  # total number of rescaling rates
    Nfreq = len(paras['frequency'])  # total number of frequency ranges

    data_sampling_rate = stream[0].stats.sampling_rate
    for itr in stream:
        assert(itr.stats.sampling_rate == data_sampling_rate)

    # load ML models
    phasemodels = []
    for imodel in paras['MLmodel']:  # loop over each model id
        for irescaling in paras['rescaling']:  # loop over each rescaling_rate
            phasemodels.append(load_MLmodel(model_id=imodel, rescaling_rate=irescaling, 
                                            overlap_ratio=paras['overlap_ratio'], blinding=None))  # blinding=(0, 0)
    
    # check
    assert(len(phasemodels)==Nmlmd*Nresc)

    # apply model to data streams
    probs_all = []
    for kmodel in phasemodels:
        for ifreq in paras['frequency']:
            stream_ft = stream.copy()
            if (isinstance(ifreq, (list))):
                # filter data in specified frequency range
                stfilter(stream_ft, fband=ifreq)                

            # obtain phase probability for each model and frequency
            probs_all.append(kmodel.annotate(stream=stream_ft))
            del stream_ft
    
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

    output = {}
    if (paras['output'].lower() == 'prob') or (paras['output'].lower() == 'all'):
        output['prob'] = prob

    if (paras['output'].lower() == 'pick') or (paras['output'].lower() == 'all'):
        # get picks   
        output['pick'] = _get_picks(prob=prob, paras=paras)

        # # check
        # pick_check = phasemodels[0].classify(stream, P_threshold=paras['pick']['P_threshold'],
        #                                      S_threshold=paras['pick']['S_threshold'])
        # assert(len(output['pick'])==len(pick_check.picks))
        # for hh, hhpick in enumerate(output['pick']):
        #     print(pick_check.picks[hh], hhpick)
        #     print()
        #     print(pick_check.picks[hh].__dict__, hhpick.__dict__)
        #     print()
        #     assert(pick_check.picks[hh].__dict__ == hhpick.__dict__)
        #     print('they are the same')

    return output


def _get_picks(prob, paras):
    # get picks from probabilities
    
    phase_tags = ['P', 'S']
    picks = sbu.PickList()
    for itag in phase_tags:
        if isinstance(paras['pick'][f"{itag}_threshold"],(float,int)):
            # use a picking threshold 
            picks += WaveformModel.picks_from_annotations(annotations=prob.select(channel=f"*_{itag}"),
                                                          threshold=paras['pick'][f"{itag}_threshold"],
                                                          phase=itag)
        elif (paras['pick'][f"{itag}_threshold"].lower()=='max'):
            # make a pick anyway, recoginze the maximum probability as the threshold
            assert(prob.select(channel=f"*_{itag}").count() == 1)  # should be only one trace
            iprob = prob.select(channel=f"*_{itag}")[0]  # phase probabilities
            iprob_max = np.abs(iprob.max())  # maximum phase probability, treat it as the picking threshold
            trace_id = f"{iprob.stats.network}.{iprob.stats.station}.{iprob.stats.location}"  # the trace id
            ipick_time = iprob.stats.starttime + iprob.times()[iprob.data.argmax()]  # the picking time of this phase
            ipick = sbPick(trace_id=trace_id, start_time=ipick_time, end_time=ipick_time, 
                           peak_time=ipick_time, peak_value=iprob_max, phase=itag)  # take the time at the maximum probability as the pick time
            picks += ipick
        else:
            raise ValueError

    return sbu.PickList(sorted(picks))


