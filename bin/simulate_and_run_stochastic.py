import matplotlib
matplotlib.use("agg")
from schumann.classes import (MagTimeSeries, MagSpectrum, Params,
                              powerlaw_transfer_function)

from schumann.classes.coarseGrain import coarseGrain
from gwpy.timeseries import TimeSeriesDict, TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.spectrogram import Spectrogram
import optparse
import schumann.utils as schutils
import numpy as np

from datetime import date

def parse_command_line():
    parser = optparse.OptionParser()
    parser.add_option("--ini-file", "-i",
                      help="param file", default=None,
                      type=str, dest='param_file')
    params, args = parser.parse_args()
    return params

def get_params(param_file):
    params = Params.from_config(param_file)
    return params

def generate_gauss_ifo_data(ifoparams, generalparams):
    # generate gaussian data
    asd1 = MagSpectrum.read(ifoparams['gw_noise_asd'])
    if asd1.f0.value != 0:
        raise ValueError('Initial frequency of supplied ASDs must be zero')

    # get psd
    psd1 = MagSpectrum(asd1.value**2, df=asd1.df, f0=asd1.f0, name=
                      ifoparams['name']+':gauss_data')
    fname = (generalparams['output_prefix'] + '/' + 'gaussian_frames/'
             + '%s-GAUSSIAN-DATA-%d-DAYS.gwf')
    ndays = float(generalparams['ndays'])
    # generate gaussian noise
    print('Generating noise for %s:' % ifoparams['name'])
    print('Writing it to ' + fname % (ifoparams['name'], float(ndays)))
    print('============================')
    ts1 = psd1.generate_gaussian_timeseries(float(ndays) * 86400,
                                           float(generalparams['sample_rate']),
                                           name=psd1.name)
    ts1.write(str(fname % (ifoparams['name'], ndays)))  # , overwrite=True)
    ts1_read = TimeSeries.read(str(fname % (ifoparams['name'], ndays)), ifoparams['name']+':gauss_data')
    plot = ts1.asd(10).plot()
    ax = plot.gca()
    ax.plot(asd1)
    ax.plot(ts1_read.asd(10))
    plot.savefig('gauss_data_plot_test')
    print ('>>>>>>> DONE GENERATING GAUSSIAN NOISE')


def get_magnetic_noise_matfiles(ifoparams, generalparams):
    from datetime import datetime, timedelta

    # number of days
    mdaystart = datetime.strptime(generalparams['mag_day_start'], '%Y%m%d')
    ndays = float(generalparams['ndays'])

    # load gaussian data
    gauss_fname = (generalparams['output_prefix'] + '/' + 'gaussian_frames/'
             + '%s-GAUSSIAN-DATA-%d-DAYS.gwf')
    print('Loading Gaussian Data')
    gauss1 = MagTimeSeries.read(str(gauss_fname % (ifoparams['name'], float(ndays))),
                                ifoparams['name']+':gauss_data')
    # loop over days
    print('Loading magnetic data')
    for day in range(int(ndays)):
        magfilename = (mdaystart + timedelta(day)).strftime('%Y%m%d')
        magfile1 = ifoparams['mag_directory'] + magfilename + '.mat'
        # construct mag time series
        if day == 0:
            magts1 = MagTimeSeries.from_coughlin_matfile(magfile1,
                                                         name='%s:mag_data' % ifoparams['name'])
        else:
            magts1 = magts1.append(MagTimeSeries.from_coughlin_matfile(magfile1,
                                                                       name='%s:mag_data' % ifoparams['name']),
                                                                       inplace=False, gap='pad')
    # get and apply transfer functions to mag 1
    print('Applying transfer function to data 1')
    # do it on 1 hour timescales
    st = magts1.times[0].value
    for ii in range(int(24 * ndays)):
        print('\t\t Running hour %d' % (ii + 1))
        # crop to our time
        magfft1 = magts1.crop(st, st+3600).do_unnormalized_fft()
        # get transfer function
        tf1 = powerlaw_transfer_function(float(ifoparams['kappa'])*1e-23,
                                    float(ifoparams['beta']),
                                    magfft1.frequencies.value)
        # apply transfer function and append or initialize data
        if ii == 0:
            mag_gw_data1 = magfft1.apply_transfer_function(tf1).do_proper_ifft(magts1.epoch.value,
                                                                               name='%s:mag_data' % ifoparams['name'])
        else:
            mag_gw_data1_tmp = magfft1.apply_transfer_function(tf1).do_proper_ifft(magts1.epoch.value + 3600*ii)
            mag_gw_data1 = mag_gw_data1.append(mag_gw_data1_tmp, inplace=False)
    # add some extra zeros if mag data is short
    # needed for POL magnetometers being 1 second short
    if mag_gw_data1.size != gauss1.size:
        # add a warning, as this could be something people don't want
        print("WARNING: WE ARE PADDING THE MAGNETIC DATA WITH ZEROS")
        mag_gw_data1 = mag_gw_data1.pad((0,  gauss1.size - mag_gw_data1.size)).copy()
        magts1 = magts1.pad((0, gauss1.size - magts1.size)).copy()


    # add magnetic and GW data together
    final_data_1 = TimeSeries((mag_gw_data1.value + gauss1.value),
                              sample_rate=gauss1.sample_rate,
                              unit=gauss1.unit, name='%s:FAKE-CONTAMINATED' % ifoparams['name'],
                              channel='%s:FAKE-CONTAMINATED' % ifoparams['name'],
                              epoch=magts1.times[0].value)

    dur = ndays * 86400
    st = final_data_1.times[0].value

    # write raw magnetic data to a file
    rawmag_fname = generalparams['output_prefix'] + '/correlated_mag_data/%s-RAW-MAG-DATA-%d-%d.gwf'
    magts1.write(str(rawmag_fname % (ifoparams['name'], st, dur)))


    # get total duration we've created
    combined_fname = generalparams['output_prefix'] + '/contaminated_frames/%sFAKE-CONTAMINATED-%d-%d.gwf'
    mag_fname = generalparams['output_prefix'] + '/correlated_mag_data/%sFAKE-MAG-%d-%d.gwf'
    final_data_1.write(str(combined_fname % (ifoparams['name'], st, dur)))
    mag_gw_data1.write(str(mag_fname % (ifoparams['name'], st, dur)))
    fd1_plot = final_data_1.asd(10).plot()
    ax = fd1_plot.gca()
    ax.plot(mag_gw_data1.asd(10))
    ax.plot(gauss1.asd(10))
    ax.set_xlim(1,64)
    ax.set_xscale('linear')
    fd1_plot.add_legend()
    fd1_plot.savefig('test_plot')

def generate_magnetometer_csd(ifoparams1, ifoparams2, generalparams, stochparams):
    from datetime import datetime, timedelta
    from astropy.time import Time
    # get start time and duration from our paramfile
    st = Time(datetime.strptime(generalparams['mag_day_start'], '%Y%m%d')).gps
    dur = int(float(generalparams['ndays']) * 86400)

    # general frame name
    rawmag_fname = generalparams['output_prefix'] + '/correlated_mag_data/%s-RAW-MAG-DATA-%d-%d.gwf'

    # specific frame names
    rawmag_fname1 = rawmag_fname % (ifoparams1['name'], st, dur)
    rawmag_fname2 = rawmag_fname % (ifoparams2['name'], st, dur)

    # load them up
    magts1 = MagTimeSeries.read(str(rawmag_fname1), '%s:mag_data' % ifoparams1['name'])
    magts2 = MagTimeSeries.read(str(rawmag_fname2), '%s:mag_data' % ifoparams2['name'])

    # take csd spectrogram between two magnetic timeseries
    magcsd = magts1.csd_spectrogram(magts2, float(stochparams['segdur']), fftlength=float(stochparams['segdur']),
                                    overlap=float(stochparams['segdur'])/2., nproc=8)

    # generate hdf5 file name for output
    csd_fname = generalparams['output_prefix'] + '/correlated_mag_data/%s-%s-COMPLEX-CSD-%d-%d.hdf5'

    # coarsegrain
    # note there's a gwpy bug for slicing on rows right now
    # it produces a FrequencySeries with the wrong metadata.
    # we'll fix it on the fly
    # TODO: open a GWpy issue about this problem

    # coarsegrain once to get shape
    ntimes = magcsd.shape[0]
    spec1 = coarseGrain(FrequencySeries(magcsd[0, :].value, df=magcsd.df, f0=magcsd.f0),
                        flowy=float(stochparams['deltaf']),
                        deltaFy=float(stochparams['deltaf']))

    # initialize array for speed
    # make sure to tell it you want a complex array
    # otherwise it'll automatically pick out
    # real part of what you put into it
    newarr = np.zeros((ntimes, spec1.size), dtype=complex)

    print('COARSEGRAINING MAG DATA FOR STOCHASTIC USE')

    # loop over each individual spectrum and coarse grain
    for ii in range(ntimes):
        newarr[ii, :] = coarseGrain(FrequencySeries(magcsd[ii, :].value, df=magcsd.df, f0=magcsd.f0),
                        flowy=float(stochparams['deltaf']),
                        deltaFy=float(stochparams['deltaf'])).value

    # create final spectrogram object
    final_csd = Spectrogram(newarr, df=spec1.df, f0=spec1.f0, name=magcsd.name,
                            channel=magcsd.channel, times=magcsd.times)



    # write to an hdf5 file
    final_csd.write(str(csd_fname % (ifoparams1['name'], ifoparams2['name'], st, dur)), overwrite=True)

    # add final message so you know where to look after
    print('WROTE CORRELATED SPECTROGRAM TO %s' % str(csd_fname % (ifoparams1['name'],
                                                                  ifoparams2['name'],
                                                                  st, dur)))

    # diagnostic plot
    # plot = (final_csd.real.abs()).plot(norm=matplotlib.colors.LogNorm(vmin=1e-7, vmax=1e-5), cmap='plasma')
    # ax = plot.gca()
    # ax.set_ylim(5, 30)
    # plot.add_colorbar()
    # ax.set_title('$M(f)$ for this search')
    # plot.savefig('test_csd_spectrogram')

if __name__ == "__main__":
    cli_args = parse_command_line()
    params = get_params(cli_args.param_file)
    schutils.setup_sim_directory(params.general['output_prefix'])

    # general gaussian noise for both detectors
    generate_gauss_ifo_data(params.ifo1, params.general)
    generate_gauss_ifo_data(params.ifo2, params.general)

    # create correlated magnetic data for both detectors
    get_magnetic_noise_matfiles(params.ifo1, params.general)
    get_magnetic_noise_matfiles(params.ifo2, params.general)

    # generate complex csd between two magnetometers
    generate_magnetometer_csd(params.ifo1, params.ifo2, params.general, params.stochastic)
