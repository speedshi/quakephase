

import seisbench.models as sbm



def load_MLmodel(model_id, rescaling_rate=None, overlap_ratio=0.5, blinding=None):
    '''
    NOTE sepration for model_id should be ','
    '''

    model_idsp = model_id.split(',')

    if model_idsp[0].lower() == 'seisbench':
        # use seisbench phase engine
        xmodel = _sbmodel(ML_model=model_idsp[1].lower(),
                          pre_trained=model_idsp[2].lower(),
                          rescaling_rate=rescaling_rate,
                          overlap_ratio=overlap_ratio,
                          blinding=blinding)
    else:
        # other supported ML engines and plateforms to be added...
        raise ValueError(f"Unrecognize model_id: {model_id}!")

    return xmodel



# load seisbench ML model-------------------------------------------------------------------------
def _sbmodel(ML_model, pre_trained, rescaling_rate=None, overlap_ratio=None, blinding=None):
    '''
    Load and config seisbench ML models.

    INPUT:
        ML_model: str,
            which SeisBench ML model to use,
            can be 'EQT', 'PNT', 'GPD', 'BasicPhaseAE', 'PhaseNetLight'.
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
        blinding: list of int or None,
            set no missing prediction points at the earliest and last of prediction windows, 
            only work for EQT and PNT?
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
    elif (ML_model.upper() == 'BASICPHASEAE'):
        sbmodel = sbm.BasicPhaseAE.from_pretrained(pre_trained.lower(), version_str="latest")
    elif (ML_model.upper() == 'PHASENETLIGHT'):
        sbmodel = sbm.PhaseNetLight.from_pretrained(pre_trained.lower(), version_str="latest")
    else:
        raise ValueError('Input SeisBench model name: {} unrecognized!'.format(ML_model))

    # rescaling the model
    if (rescaling_rate is not None) and (isinstance(rescaling_rate,(int,float))):
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
        assert(isinstance(overlap_ratio,(int,float)))  # should be float or int number
        sbmodel.default_args['overlap'] = int(sbmodel.in_samples * overlap_ratio)

    if blinding is not None:
        # set no missing prediction points at the earliest and last of prediction windows, only work for EQT and PNT?
        sbmodel.default_args['blinding'] = blinding  

    return sbmodel
#----------------------------------------------------------------------------------------------------------------




