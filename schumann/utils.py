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
                'webpage']
    for dirname in dirnames:
        safe_create_directory(output_pref + '/' + dirname)