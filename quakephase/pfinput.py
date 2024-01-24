"""
Load parameter file.
Check input parameters.
"""


import yaml



def load_check_input(file_para):

    # load paramters
    with open(file_para, 'r') as file:
        paras = yaml.safe_load(file)

    # check 'MLmodel' setting
    assert(isinstance(paras['MLmodel'], (list,)))

    # check 'overlap_ratio' setting
    if isinstance(paras['overlap_ratio'],(int,float)) and (0<=paras['overlap_ratio']<1):
        pass
    else:
        raise ValueError(f"Invalid input for overlap_ratio. Must be a value in range [0, 1)")

    # check 'rescaling' setting
    assert(isinstance(paras['rescaling'], (list,)))
    for jj in range(len(paras['rescaling'])):
        if isinstance(paras['rescaling'][jj], (int,float)):
            if paras['rescaling'][jj] <= 0:
                raise ValueError(f"rescaling should larger than 0. Current input is {paras['rescaling'][jj]}!")
        elif isinstance(paras['rescaling'][jj],(str)) and (paras['rescaling'][jj].lower()=='none'):
            paras['rescaling'][jj] = None
        else:
            raise ValueError(f"Invalid input for rescaling: {paras['rescaling'][jj]}!")

    # check 'frequency' setting
    assert(isinstance(paras['frequency'], (list,)))
    for ii, ifreq in enumerate(paras['frequency']):
        if isinstance(ifreq, (str,)): 
            assert(ifreq.lower()=='none')
            paras['frequency'][ii] = None
        elif isinstance(ifreq, (list,)):
            assert(len(ifreq)==2)
            for jjf in range(len(ifreq)):
                if isinstance(ifreq[jjf], (int,float)):
                    pass
                elif isinstance(ifreq[jjf],(str)) and (ifreq[jjf].lower()=='none'):
                    paras['frequency'][ii][jjf] = None
                else:
                    raise ValueError(f"Invalid input for frequency paramter: {ifreq[jj]}!")
        else:
            raise ValueError(f"Invalid input for frequency paramter: {ifreq}!")

    # check 'output' setting
    if paras['output'].lower() not in ['prob', 'pick', 'all']:
        raise ValueError(f"Unrecognized output type {paras['output']}!")
    
    # check 'pick' setting
    if 'pick' in paras:
        if 'format' not in paras['pick']:
            paras['pick']['format'] = None
        elif paras['pick']['format'].lower() == 'none':
            paras['pick']['format'] = None
        elif paras['pick']['format'].lower() not in ['dataframe', 'dict', 'list']:
            raise ValueError(f"Unrecognized pick format {paras['pick']['format']}!")

        if paras['pick']['method'].lower() not in ['threshold', 'peak', 'max']:
            raise ValueError(f"Unrecognized pick method {paras['pick']['method']}!")

        if paras['pick']['method'].lower() in ['threshold', 'peak']:
            for itag in ['P_threshold', 'S_threshold']:
                if isinstance(paras['pick'][itag],(int,float)):
                    pass
                elif (isinstance(paras['pick'][itag],str)) and (paras['pick'][itag].lower()=='none'):
                    paras['pick'][itag] = None
                else:
                    raise ValueError(f"Invalid input for pick_{itag}: {paras['pick'][itag]}!")
        
        if paras['pick']['method'] == 'peak':
            if 'nb_threshold' not in paras['pick']:
                paras['pick']['nb_threshold'] = None
            elif isinstance(paras['pick']['nb_threshold'],str) and (paras['pick']['nb_threshold'].lower()=='none'):
                paras['pick']['nb_threshold'] = None
            elif isinstance(paras['pick']['nb_threshold'],(int,float)):
                pass
            elif isinstance(paras['pick']['nb_threshold'],list):
                assert(len(paras['pick']['nb_threshold'])==2)
                to_compare = True
                for kk,kkzz in enumerate(paras['pick']['nb_threshold']):
                    if isinstance(kkzz,(int,float)):
                        pass
                    elif isinstance(kkzz,str) and (kkzz.lower()=='none'):
                        paras['pick']['nb_threshold'][kk] = None
                        to_compare = False
                    else:
                        raise ValueError(f"Invalid input for pick_nb_threshold: {kkzz}!")
                if to_compare: assert(paras['pick']['nb_threshold'][0]<=paras['pick']['nb_threshold'][1])
            else:
                raise ValueError(f"Invalid input for pick_nb_threshold : {paras['pick']['nb_threshold']}!")

            if 'distance' not in paras['pick']:
                paras['pick']['distance'] = None
            elif isinstance(paras['pick']['distance'],str) and (paras['pick']['distance'].lower()=='none'):
                paras['pick']['distance'] = None
            elif isinstance(paras['pick']['distance'],int):
                assert(paras['pick']['distance']>=1)
            else:
                raise ValueError(f"Invalid input for pick_distance: {paras['pick']['distance']}!")

            if 'prominence' not in paras['pick']:
                paras['pick']['prominence'] = None
            elif isinstance(paras['pick']['prominence'],str) and (paras['pick']['prominence'].lower()=='none'):
                paras['pick']['prominence'] = None
            elif isinstance(paras['pick']['prominence'],(int,float)):
                pass
            elif isinstance(paras['pick']['prominence'],list):
                assert(len(paras['pick']['prominence'])==2)
                to_compare = True
                for kk,kkzz in enumerate(paras['pick']['prominence']):
                    if isinstance(kkzz,(int,float)):
                        pass
                    elif isinstance(kkzz,str) and (kkzz.lower()=='none'):
                        paras['pick']['prominence'][kk] = None
                        to_compare = False
                    else:
                        raise ValueError(f"Invalid input for pick_prominence: {kkzz}!")
                if to_compare: assert(paras['pick']['prominence'][0]<=paras['pick']['prominence'][1])
            else:
                raise ValueError(f"Invalid input for pick_prominence: {paras['pick']['prominence']}!")

            if 'width' not in paras['pick']:
                paras['pick']['width'] = None
            elif isinstance(paras['pick']['width'],str) and (paras['pick']['width'].lower()=='none'):
                paras['pick']['width'] = None
            elif isinstance(paras['pick']['width'],int):
                assert(paras['pick']['width']>=0)
            elif isinstance(paras['pick']['width'],list):
                assert(len(paras['pick']['width'])==2)
                to_compare = True
                for kk,kkzz in enumerate(paras['pick']['width']):
                    if isinstance(kkzz,(int)):
                        assert(kkzz>=0)
                    elif isinstance(kkzz,str) and (kkzz.lower()=='none'):
                        paras['pick']['width'][kk] = None
                        to_compare = False
                    else:
                        raise ValueError(f"Invalid input for pick_width: {kkzz}!")
                if to_compare: assert(paras['pick']['width'][0]<=paras['pick']['width'][1])
            else:
                raise ValueError(f"Invalid input for pick_width: {paras['pick']['width']}!")

            if 'wlen' not in paras['pick']:
                paras['pick']['wlen'] = None
            elif isinstance(paras['pick']['wlen'],str) and (paras['pick']['wlen'].lower()=='none'):
                paras['pick']['wlen'] = None
            elif isinstance(paras['pick']['wlen'],int):
                assert(paras['pick']['wlen']>=0)
            else:
                raise ValueError(f"Invalid input for pick_wlen: {paras['pick']['wlen']}!")
            
            if 'rel_height' not in paras['pick']:
                paras['pick']['rel_height'] = 0.5
            elif isinstance(paras['pick']['rel_height'],(int,float)):
                assert(0<=paras['pick']['rel_height']<=1)
            else:
                raise ValueError(f"Invalid input for pick_rel_height: {paras['pick']['rel_height']}!")

            if 'plateau_size' not in paras['pick']:
                paras['pick']['plateau_size'] = None
            elif isinstance(paras['pick']['plateau_size'],str) and (paras['pick']['plateau_size'].lower()=='none'):
                paras['pick']['plateau_size'] = None
            elif isinstance(paras['pick']['plateau_size'],int):
                assert(paras['pick']['plateau_size']>=0)
            elif isinstance(paras['pick']['plateau_size'],list):
                assert(len(paras['pick']['plateau_size'])==2)
                to_compare = True
                for kk,kkzz in enumerate(paras['pick']['plateau_size']):
                    if isinstance(kkzz,(int)):
                        assert(kkzz>=0)
                    elif isinstance(kkzz,str) and (kkzz.lower()=='none'):
                        paras['pick']['plateau_size'][kk] = None
                        to_compare = False
                    else:
                        raise ValueError(f"Invalid input for pick_plateau_size: {kkzz}!")
                if to_compare: assert(paras['pick']['plateau_size'][0]<=paras['pick']['plateau_size'][1])
            else:
                raise ValueError(f"Invalid input for pick_plateau_size: {paras['pick']['plateau_size']}!")

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

    # check 'data' setting
    if 'data' not in paras: paras['data'] = {}
    if 'component_input' in paras['data']:
        if (len(paras['data']['component_input'])<=3):
            pass
        else:
            raise ValueError(f"Invalid input for data_component_input: {paras['data']['component_input']}! Must lesst than 3 components!")

    if 'auto_expend' in paras['data']:
        if 'method' not in paras['data']['auto_expend']:
            raise ValueError(f"Need to specify data_auto_expend_method!")
        elif type(paras['data']['auto_expend']['method']) is str:
            # automatically expend data if the input data duration is shorter than required
            # expend using input samples at the beginning and end of the data
            pass
        else:
            raise ValueError(f"Invalid input for data_auto_expend_method: {paras['data']['auto_expend']['method']}!")

        if 'window_ratio' not in paras['data']['auto_expend']:
            paras['data']['auto_expend']['window_ratio'] = 1.0
        elif isinstance(paras['data']['auto_expend']['window_ratio'], (int,float)):
            # accept float or int
            # the ratio of the expended window size to the required input duration, 1.0 means the same size
            # 2.0 means the final expended window is 2 times of the required input duration
            pass
        else:
            raise ValueError(f"Invalid input for data_auto_expend_window_ratio: {paras['data']['auto_expend']['window_ratio']}!")
    
    return paras




