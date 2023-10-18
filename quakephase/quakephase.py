

import yaml
from .load_MLmodel import load_MLmodel
from .qkprocessing import stfilter, prob_ensemble
import seisbench.util as sbu
from seisbench.models.base import WaveformModel



def _check_input(paras):

    assert(isinstance(paras['MLmodel'], (list,)))
    assert(isinstance(paras['rescaling'], (list,)))
    assert(isinstance(paras['frequency'], (list,)))

    for ifreq in paras['frequency']:
        if isinstance(ifreq, (str,)): 
            assert(ifreq.lower()=='none')
        elif isinstance(ifreq, (list,)):
            assert(len(ifreq)==2)
        else:
            raise ValueError(f"Invalid input for frequency paramter: {ifreq}!")

    if paras['output'].lower() not in ['prob', 'pick', 'all']:
        raise ValueError(f"Unrecognized output type {paras['output']}!")

    return


def qkphase(stream, file_para='parameters.yaml'):
    '''
    INPUT:
        stream: three-component obspy stream object;
        file_para: str, path to the paramter YAML file;

    OUTPUT:

    '''

    # load input parameters
    with open(file_para, 'r') as file:
        paras = yaml.safe_load(file)

    # check inputs
    _check_input(paras=paras)

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
        phase_tags = ['P', 'S']
        picks = sbu.PickList()
        for itag in phase_tags:
            picks += WaveformModel.picks_from_annotations(annotations=prob.select(channel=f"*_{itag}"),
                                                          threshold=paras['pick'][f"{itag}_threshold"],
                                                          phase=itag)
        output['pick'] = sbu.PickList(sorted(picks))

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



