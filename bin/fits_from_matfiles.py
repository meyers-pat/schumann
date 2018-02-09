import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from schumann.classes import CoughlinMagFile, SchumannParamTable
from glob import glob
import optparse
import numpy as np
from astropy.io import ascii
from pandas.plotting import scatter_matrix


def parse_command_line():
    """
    parse command line
    """
    parser = optparse.OptionParser()
    parser.add_option('--magdir', help='matfile directory', default=None,
                      dest='magdir')
    params, args = parser.parse_args()
    return params


def run_and_fit_mags(args):
    peaks = np.asarray([7.25, 13.98, 20.0, 26.3])
    # set initial Q values
    Qs = np.asarray([6, 5.8, 5.0, 10])
    # set initial amplitudes
    amps = np.asarray([1, 0.8, 0.25, 0.2])
    files = glob(args.magdir + '*.mat')
    First = True
    for eachfile in files:
        myfile = CoughlinMagFile(eachfile, 'r')
        st = myfile.start_time
        # for each 2 hours
        for ii in range(12):
            spec = myfile.average_spectrum_cropped(st, st + 7200)
            spec = spec.crop(4, 30).coarse_grain(5, 0.25)
            myfit, init = spec.fit(peaks, Qs, amps * spec.max().value)
            if First:
                # initialize table
                mytab = SchumannParamTable.from_fit(myfit.parameters, st)
                First = False
            else:
                # add to table
                mytab.add_fit(myfit.parameters, st)
            # increment start time
            st += 7200
    ascii.write(mytab, 'fit_table.dat')
    scatter_matrix((mytab['peak0', 'amp0', 'starttime']).to_pandas(),
                   diagonal='kde')
    plt.tight_layout()
    plt.savefig('scatter_matrix_plot')


if __name__ == "__main__":
    args = parse_command_line()
    run_and_fit_mags(args)
