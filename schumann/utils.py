import os

def safe_create_directory(dirname):
    """
    safely create directories
    Parameters
    ----------
    dirname : `str`
        name of directroy to create

    Returns
    -------
    """
    try:
        os.makedirs(dirname)
    except:
        print('Directory %s already exists' % dirname)


def setup_sim_directory(output_pref):
    """
    setup simulation directory structure
    Parameters
    ----------
    output_pref : `str`
        output file prefix

    Returns
    -------
    """
    dirnames = ['gaussian_frames',
                'contaminated_frames',
                'stochastic_output',
                'chains',
                'correlated_mag_data',
                'webpage',
                'stochastic_files',
                'stochastic_files/cache']
    for dirname in dirnames:
        safe_create_directory(output_pref + '/' + dirname)

def setup_stochastic_params(stochparams, ifoparams1, ifoparams2,
        generalparams):
    #shorter
    pref = generalparams['output_prefix']
    stochparams['output_prefix_full'] = pref + '/stochastic_output/stoch'
    stochparams['cachepath'] = pref + '/stochastic_files/cache/'
    stochparams['ifo1'] = ifoparams1['name']
    stochparams['ifo2'] = ifoparams2['name']
    stochparams['sample_rate'] = generalparams['sample_rate']
    stochparams['mindataload'] = int(stochparams['segdur']) * 3 + 20
    try:
        stochparams['flow']
    except:
        stochparams['flow'] = 10
    try:
        stochparams['fhigh']
    except:
        stochparams['fhigh'] = (int(stochparams['sample_rate']) / 2.) - float(stochparams['deltaf'])

    return stochparams



