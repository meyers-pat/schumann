from matplotlib import use
use('agg')
from unittest import TestCase
from gwpy.spectrogram import Spectrogram
from schumann.classes import CoughlinMagFile, SchumannParamTable
import numpy.testing as npt
import numpy as np

FILENAME = 'schumann/tests/col/736667-736668.mat'
FILENAME2 = 'schumann/tests/pol/736667-736668.mat'
DELTAF = 1. / 128
SEGDUR = 128
F0 = 0
STARTTIME = 1164758417


class TestCoughlinMagFile(TestCase):

    __all__ = ['test_file_contents', 'clipped_avg_spectrum']

    def test_cropped_avg_spectrum(self):
        testfile = CoughlinMagFile(FILENAME)
        spec = testfile.average_spectrum_cropped(testfile.start_time,
                                                 testfile.start_time + 7200)
        spec = None

    def test_cross_corr(self):
        col = CoughlinMagFile(FILENAME)
        pol = CoughlinMagFile(FILENAME2)
        xcorr = col.cross_corr(pol)
        phase = np.angle(xcorr.value)
        phase = Spectrogram(phase, dt=xcorr.dx, t0=xcorr.x0,
                            df=xcorr.df, f0=xcorr.f0,
                            unit='radians')
        plot = phase.plot(cmap='Spectral_r', vmin=-np.pi, vmax=np.pi)
        plot.savefig('test_phase_plot')
        specvar = phase.variance()
        plot = specvar.plot()
        ax = plot.gca()
        ax.set_yscale("linear")
        ax.set_xscale('linear')
        ax.set_ylim(-np.pi, np.pi)
        plot.savefig('specvar_test')


    def test_file_contents(self):
        # test that the contents of the file
        # are what they should be
        testfile = CoughlinMagFile(FILENAME)
        npt.assert_equal(testfile.df, DELTAF)
        npt.assert_equal(testfile.segdur, SEGDUR)
        npt.assert_equal(testfile.f0, F0)
        npt.assert_equal(testfile.start_time, STARTTIME)
        print testfile.cont_times
        print testfile.cont_freqs
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


class TestSchumannTable(TestCase):
    def test_schumann_table(self):
        testfile = CoughlinMagFile(FILENAME)
        spec = testfile.average_spectrum.crop(4, 30).coarse_grain(6, 0.2)
        # set initial peak values
        peaks = [7.25, 13.98, 20.0, 26.3]
        # set initial Q values
        Qs = [6, 5.8, 5.0, 10]
        # set initial amplitudes
        amps = np.asarray([1, 0.8, 0.25, 0.2]) * spec.max().value
        myfit, initial = spec.fit(peaks, Qs, amps)
        params = SchumannParamTable.from_fit(myfit.parameters,
                                             testfile.start_time)
        params.add_fit(myfit.parameters, testfile.start_time)
        print type(params['peak0'])
