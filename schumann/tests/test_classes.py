from __future__ import print_function
from matplotlib import use
use('agg')
from unittest import TestCase
from gwpy.spectrogram import Spectrogram
from schumann.classes import (CoughlinMagFile, SchumannParamTable,
                              MagTimeSeries, MagSpectrum,
                              powerlaw_transfer_function)
import numpy.testing as npt
import numpy as np

FILENAME = 'schumann/tests/col/736667-736668.mat'
FILENAME2 = 'schumann/tests/pol/736667-736668.mat'
TSFNAME = 'schumann/tests/col/20170831.mat'
TSTESTFILE = 'schumann/tests/data/test_ts.hdf5'
DELTAF = 1. / 128
SEGDUR = 128
F0 = 0
STARTTIME = 1164758417
# add two timeseries with peaks
# of same amplitude at 22 and 44 Hz
TESTMTS = MagTimeSeries(np.sin(2 * np.pi * 22 * np.arange(10000) / 100), sample_rate=100, t0=0)
TESTMTS2 = MagTimeSeries(np.sin(2 * np.pi * 44 * np.arange(10000) / 100), sample_rate=100, t0=0)
TESTMTS += TESTMTS2
DIAGNOSTIC_PLOTDIR = 'schumann/tests/plots/'
DIAGNOSTIC_DATADIR = 'schumann/tests/plots/'



class TestCoughlinMagFile(TestCase):

    def test_cropped_avg_spectrum(self):
        testfile = CoughlinMagFile(FILENAME)
        spec = testfile.average_spectrum_cropped(testfile.start_time,
                                                 testfile.start_time + 7200)
        spec = None

    # def test_cross_corr(self):
    #     col = CoughlinMagFile(FILENAME)
    #     pol = CoughlinMagFile(FILENAME2)
    #     xcorr = col.cross_corr(pol)
    #     phase = np.angle(xcorr.value)
    #     phase = Spectrogram(phase, dt=xcorr.dx, t0=xcorr.x0,
    #                         df=xcorr.df, f0=xcorr.f0,
    #                         unit='radians')
    #     plot = phase.plot(cmap='Spectral_r', vmin=-np.pi, vmax=np.pi)
    #     plot.savefig(DIAGNOSTIC_PLOTDIR + 'test_phase_plot')
    #     specvar = phase.variance()
    #     plot = specvar.plot()
    #     ax = plot.gca()
    #     ax.set_yscale("linear")
    #     ax.set_xscale('linear')
    #     ax.set_ylim(-np.pi, np.pi)
    #     plot.savefig(DIAGNOSTIC_PLOTDIR + 'specvar_test')


    def test_file_contents(self):
        # test that the contents of the file
        # are what they should be
        testfile = CoughlinMagFile(FILENAME)
        npt.assert_equal(testfile.df, DELTAF)
        npt.assert_equal(testfile.segdur, SEGDUR)
        npt.assert_equal(testfile.f0, F0)
        npt.assert_equal(testfile.start_time, STARTTIME)
        testfile.spectrogram


class TestMagSpectrum(TestCase):
    def test_fit(self):
        # test that the contents of the file
        # are what they should be
        testfile = CoughlinMagFile(FILENAME)
        spec = testfile.average_spectrum.crop(4, 30).coarse_grain(6, 0.2)
        # set initial peak values
        peaks = [7.25, 13.98, 20.0, 26.3]
        # set initial Q values
        Qs = [6, 5.8, 5.0, 10]
        # set initial amplitudes
        amps = np.asarray([1, 0.8, 0.25, 0.2]) * spec.max().value
        myfit, initial = spec.fit(peaks, Qs, amps)
        plot = spec.plot_fit(myfit)
        ax = plot.gca()
        ax.plot(spec.frequencies.value,
                initial(spec.frequencies.value))
        ax.set_xscale('linear')

    def test_generate_gaussian_timeseries(self):
        psd = MagSpectrum(TESTMTS.psd())
        psd = MagSpectrum(np.ones(100), frequencies=np.arange(100))
        psd2 = MagSpectrum(TESTMTS.psd(10, window='hann', overlap=5))
        # 10s, srate is 100
        gauss_data = psd.generate_gaussian_timeseries(10000, 100)
        print(gauss_data)
        # gauss_data = psd.generate_gaussian_timeseries(432000, 100)
        gauss_data.write(DIAGNOSTIC_DATADIR + 'test_out.gwf')
        # gauss_psd = gauss_data.psd(10, window='hann', overlap=5)

        # DIAGNOSTIC PLOTS
        # spec = gauss_data.spectrogram(303, fftlength=101, nproc=8)
        # plot = spec.plot()
        # plot.savefig(DIAGNOSTIC_PLOTDIR + 'spectrogram_gauss_data')
        # plot.close()
        # OTHER PLOT
        # plot = psd2.plot(label='original', linewidth=2, alpha=0.5)
        # ax = plot.gca()
        # ax.plot(gauss_psd, label='gaussian generated', linewidth=2, alpha=0.5)
        # ax.set_ylim(1e-2, 10)
        # ax.set_xlim(20, 24)
        # ax.set_xscale('linear')
        # plot.add_legend()
        # plot.savefig(DIAGNOSTIC_PLOTDIR + 'gauss_data_test')
        # print(psd2.sum() * psd2.df)
        # print(gauss_psd.sum() * gauss_psd.df)


class TestSchumannTable(TestCase):
    def test_schumann_table(self):
        testfile = CoughlinMagFile(FILENAME)
        spec = testfile.average_spectrum.crop(4, 30).coarse_grain(6, 0.2)
        # set initial peak values
        peaks = [7.25, 13.98, 20.0, 26.3]
        # set initial Q values
        qs = [6, 5.8, 5.0, 10]
        # set initial amplitudes
        amps = np.asarray([1, 0.8, 0.25, 0.2]) * spec.max().value
        myfit, initial = spec.fit(peaks, qs, amps)
        params = SchumannParamTable.from_fit(myfit.parameters,
                                             testfile.start_time)
        params.add_fit(myfit.parameters, testfile.start_time)


class TestMagTimseries(TestCase):
    def test_fetch_lemi(self):
        data = MagTimeSeries.fetch_lemi('H1', 'Y', 'August 15 2017', 'August 15 2017 00:00:10')
        # check that it's the correct type
        # I had issues before getting fetch
        # to return an instance of the new class
        assert(type(data) == MagTimeSeries)

    def test_fft_method(self):
        myfft = TESTMTS.do_unnormalized_fft()
        remade = myfft.do_proper_ifft(myfft.epoch)
        asd_rmade = remade.asd()
        # DIAGNOSTIC PLOTS
        #plot = (TESTMTS.asd()).plot(linewidth=3)
        #ax = plot.gca()
        #ax.plot(remade.asd(), linewidth=2)
        #ax.set_xscale('linear')
        #plot.savefig(DIAGNOSTIC_PLOTDIR + 'test_fft_ifft_compare')

    def test_apply_transfer(self):
        myfft = TESTMTS.do_unnormalized_fft()
        # create powerlaw transfer function
        tf = powerlaw_transfer_function(1, 1,
                                        myfft.frequencies.value)
        # apply it to the unnormalized fft
        myfft = myfft.apply_transfer_function(tf).copy()
        # take ifft of transfer function-ed fft
        remade = myfft.do_proper_ifft(myfft.epoch)
        # take asd and sort it
        asd = remade.asd()
        sorted = np.sort(asd.value)
        # two peaks should be factor of 2 different
        npt.assert_almost_equal(2 * sorted[-2], sorted[-1])
        #### DIAGNOSTIC PLOTS
        # asd_rmade = remade.asd()
        # plot = (TESTMTS.asd()).plot(linewidth=3)
        # ax = plot.gca()
        # ax.plot(remade.asd(), linewidth=2)
        # ax.set_ylim(1e-3, 10)
        # ax.set_xscale('linear')
        # plot.savefig(DIAGNOSTIC_PLOTDIR + 'test_apply_tf')

    def test_spectrum_methods(self):
        asd = TESTMTS.magasd(1)
        assert(type(asd) == MagSpectrum)
        fft = TESTMTS.magfft()
        assert(type(fft) == MagSpectrum)
        psd = TESTMTS.magpsd()
        assert(type(psd) == MagSpectrum)
        csd = TESTMTS.magcsd(TESTMTS)
        assert(type(csd) == MagSpectrum)

    def test_from_matfile(self):
        dat = MagTimeSeries.from_coughlin_matfile(TSFNAME)
