

from .load_MLmodel import load_MLmodel
from .qkprocessing import stfilter, prob_ensemble
from .pfinput import load_check_input
from .xpick import get_picks



def apply(stream, file_para='parameters.yaml'):
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
        output['pick'] = get_picks(prob=prob, paras=paras)

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





