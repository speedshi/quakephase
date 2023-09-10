

import yaml
import seisbench.models as sbm


# %% This is how seisbench resample the input stream ----------------------------------
@staticmethod
def _sbresample(stream, sampling_rate):
    """
    Perform inplace resampling of stream to a given sampling rate.

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
#---------------------------------------------------------------------------------------


# %% load seisbench ML model-------------------------------------------------------------------------
def _sbmodel(ML_model, pre_trained, rescaling_rate=None, overlap_ratio=None, blinding=None):
    '''
    Load and config seisbench model.

    INPUT:
        ML_model: str,
            which SeisBench ML model to use,
            can be 'EQT', 'PNT', 'GPD'.
        pre_trained: str,
            specify which pre-trained data set of the chosen ML model,
            can be 'stead', 'scedc', 'ethz', etc.
            for detail check SeisBench documentation.
        rescale_rate: float
            specify the rescaling-rate for ML phase identification and picking,
            = used_model_sampling_rate / original_model_sampling_rate.
            None means using model default sampling rate.
        overlap_ratio: float [0, 1]
            overlap_ratio when apply to continuous data or longer data segments.
            e,g, 0.5 means half-overlapping;
            0.6 means 60% overlapping;
            0.8 means 80% overlapping;
            0.0 means no overlapping.
        blinding: list or None,
            set no missing prediction points at the earliest and last of prediction windows, only work for EQT and PNT?
            e.g. blinding = (0, 0) means no missing predictions points;
            if blinding = None, use default settings.

    OUTPUT:
        sbmodel: SeisBench model.
    '''

    # specify a ML model
    if (ML_model.upper() == 'EQT') or (ML_model.upper() == 'EQTRANSFORMER'):
        sbmodel = sbm.EQTransformer.from_pretrained(pre_trained.lower(), version_str="latest")  # , update=True, force=True, wait_for_file=True
    elif (ML_model.upper() == 'PNT') or (ML_model.upper() == 'PHASENET'):
        sbmodel = sbm.PhaseNet.from_pretrained(pre_trained.lower(), version_str="latest")  # , update=True, force=True, wait_for_file=True
    elif (ML_model.upper() == 'GPD'):
        sbmodel = sbm.GPD.from_pretrained(pre_trained.lower(), version_str="latest")  # , update=True, force=True, wait_for_file=True
    else:
        raise ValueError('Input SeisBench model name: {} unrecognized!'.format(ML_model))

    # rescaling the model
    if rescaling_rate is not None:
        sbmodel.sampling_rate = sbmodel.sampling_rate * rescaling_rate  # reset model sampling rate according to the rescaling rate

    # deactivate any default filtering, as the input data stream should already been filtered
    sbmodel.filter_args = None  # disable the default filtering
    sbmodel.filter_kwargs = None  # disable the default filtering

    # set overlapping
    if overlap_ratio is None:
        # default using 80% overlap-ratio
        KK = 5  # every point is covered by KK windows
        sbmodel.default_args['overlap'] = int(sbmodel.in_samples * (1 - 1.0/KK))
    else:
        sbmodel.default_args['overlap'] = int(sbmodel.in_samples * overlap_ratio)

    if blinding is not None:
        # set no missing prediction points at the earliest and last of prediction windows, only work for EQT and PNT?
        sbmodel.default_args['blinding'] = blinding  

    return sbmodel
#----------------------------------------------------------------------------------------------------------------




def qkphase(stream, file_para='parameters.yaml'):
    '''
    INPUT:
        stream: three-component obspy stream object;
        file_para: str, path to the paramter YAML file;

    OUTPUT:

    '''

    # load parameters
    with open(file_para, 'r') as file:
        paras = yaml.safe_load(file)

    # load ML model
    para_model = paras['MLmodel']
    phasemodels = []
    for imodel in para_model:
        if imodel.split('.')[0].lower() == 'seisbench':
            # use seisbench phase engine
            for irescale in paras['rescaling']:
                phasemodels.append(_sbmodel(ML_model=imodel.split('.')[1].lower(), 
                                            pre_trained=imodel.split('.')[2].lower(), 
                                            rescaling_rate=irescale, 
                                            overlap_ratio=paras['overlap_ratio'], 
                                            blinding=None))
        else:
            raise ValueError(f"Unrecognize phase engine: {imodel.split('.')[0]}!")

    # apply model to the stream
    if paras['output'].lower() == 'prob':
        for kmodel in phasemodels:
            prob = kmodel.annotate(stream=stream)  # obtain phase probabilies
        ### TO DO:
        ### how to aggregate results from different models
        return prob
    elif paras['output'].lower() == 'pick':
        for kmodel in phasemodels:
            pick = kmodel.classify(stream=stream, 
                                   P_threshold=paras['pick']['P_threshold'], 
                                   S_threshold=paras['pick']['S_threshold'])
        ### TO DO:
        ### how to aggregate results from different models
        return pick
    else:
        raise ValueError(f"Unrecognized output type {paras['output']}!")

    return



