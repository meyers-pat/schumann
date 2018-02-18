from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.spectrogram import Spectrogram
import numpy.testing as npt
from .coarseGrain import coarseGrain
from astropy import units
from astropy.modeling import models, fitting
from h5py import File as h5File
from datetime import date
from astropy.time import Time
from astropy.table import QTable
import os
import numpy as np

# TODO add docstrings for all methods

__all__ = ['CoughlinMagFile',
           'SchumannPeak',
           'MagSpectrum',
           'MagTimeSeries',
           'SchumannParamTable']


class CoughlinMagFile(h5File):
    """
    Info from a "coughlin data file"
    """
    def __init__(self, *args, **kwargs):
        super(CoughlinMagFile, self).__init__(*args, **kwargs)
        self.segdur = self['data']['T'].value.squeeze()
        ts = self['data']['tt'].value.squeeze()
        # get day in datetime format
        day = date.fromordinal(int(np.floor(ts[0])) - 366)
        # get seconds associated with each time
        secs = [np.round((tdate - np.floor(tdate)) * 86400, 3) for tdate in ts]
        # convert to gps using astropy
        self.times = Time(day.strftime('%Y-%m-%d')).gps + np.asarray(secs)
        self.start_time = self.times[0]
        self.frequencies = self['data']['ff'].value.squeeze()
        self.df = self.frequencies[1] - self.frequencies[0]
        self.f0 = self.frequencies[0]
        ts = self.times
        tdifs = ts[1:] - ts[:-1]
        cont_times = np.max(tdifs) == np.min(tdifs)
        self.cont_times = cont_times
        fs = self.frequencies
        fdifs = fs[1:] - fs[:-1]
        self.cont_freqs = np.max(fdifs) == np.min(fdifs)

    # @property
    # def start_time(self):
    #     # date string
        # self.times[0]

    # @property
    # def segdur(self):
    #     segment duration
    #     return self['data']['T'].value.squeeze()

    # @property
    # def times(self):
    #     ts = self['data']['tt'].value.squeeze()
    #     # get day in datetime format
    #     day = date.fromordinal(int(np.floor(ts[0])) - 366)
    #     # get seconds associated with each time
    #     secs = [np.round((tdate - np.floor(tdate)) * 86400, 3) for tdate in ts]
    #     # convert to gps using astropy
    #     ts = Time(day.strftime('%Y-%m-%d')).gps + np.asarray(secs)
    #     return ts

    # @property
    # def frequencies(self):
    #     # list of frequencies from file
    #     return self['data']['ff'].value.squeeze()

    # @property
    # def df(self):
    #     return self.frequencies[2] - self.frequencies[1]

    # @property
    # def f0(self):
    #     return self.frequencies[0]

    # @property
    # def cont_times(self):
    #     ts = self.times
    #     tdifs = ts[1:] - ts[:-1]
    #     cont_times = np.max(tdifs) == np.min(tdifs)
    #     return cont_times

    # @property
    # def cont_freqs(self):
    #     fs = self.frequencies
    #     fdifs = fs[1:] - fs[:-1]
    #     cont_freqs = np.max(fdifs) == np.min(fdifs)
    #     return cont_freqs

    @property
    def spectrogram(self):
        # check if times are continuous
        # check if freqs are continuous
        kwargs = {'name': 'magnetic data', 'channel': None,
                  'unit': units.nT}
        if self.cont_times:
            kwargs['t0'] = self.times[0]
            kwargs['dt'] = self.times[2] - self.times[1]
        else:
            kwargs['times'] = self.times
        if self.cont_freqs:
            kwargs['f0'] = self.frequencies[0]
            kwargs['df'] = self.frequencies[2] - self.frequencies[1]
        else:
            kwargs['frequencies'] = self.frequencies
        return Spectrogram(self['data']['spectra'].value.squeeze(),
                           **kwargs)

    @property
    def average_spectrum(self):
        return MagSpectrum.from_freqseries(self.spectrogram.mean(axis=0))

    def average_spectrum_cropped(self, st, et):
        return MagSpectrum.from_freqseries(self.spectrogram.crop(st, et).mean(axis=0))

    @property
    def fft_amp_spectrogram(self):
        # vals = self['data']['fftamp'].value.squeeze()
        # intermediate = vals['real'] + 1j * vals['imag']
        intermediate = self['data']['fftamp'].value.squeeze()['real'] +\
            1j * self['data']['fftamp'].value.squeeze()['imag']
        # intermediate = np.zeros((self.times.size, self.frequencies.size))

        # intermediate = np.zeros(vals.flatten().size)
        kwargs = {'name': 'magnetic data', 'channel': None,
                  'unit': units.nT}
        if self.cont_times:
            kwargs['t0'] = self.times[0]
            kwargs['dt'] = self.times[2] - self.times[1]
        else:
            kwargs['times'] = self.times
        if self.cont_freqs:
            kwargs['f0'] = self.frequencies[0]
            kwargs['df'] = self.frequencies[2] - self.frequencies[1]
        else:
            kwargs['frequencies'] = self.frequencies
        return Spectrogram(intermediate,
                           **kwargs)

    def cross_corr(self, other):
        if npt.assert_almost_equal(self.frequencies, other.frequencies):
            raise ValueError('frequencies must match')
        newmat = np.zeros((self.spectrogram.shape))
        this_specgram = self.fft_amp_spectrogram.copy()
        other_specgram = other.fft_amp_spectrogram.copy()
        final_times = []
        if self.cont_times and other.cont_times:
            et = int(min(this_specgram.times[-1].value, other_specgram.times[-1].value))
            st = int(max(this_specgram.times[0].value, other_specgram.times[0].value))
            this_specgram2 = this_specgram.crop(st, et)
            other_specgram2 = other_specgram.crop(st, et)
            newmat = this_specgram2.value * other_specgram2.value
            final_times = this_specgram2.times.value
        else:
            for ii, time in enumerate(self.times):
                # check if time appears in other specgram
                if np.isin(time, other.times):
                    # find where it appears
                    other_idx = np.where(other.times == time)[0]
                    # do cross-corr
                    newmat[ii, :] = this_specgram[ii, :].value *\
                        other_specgram[other_idx, :].value
                    # add this to final times list
                    final_times.append(time)
        # return
        return Spectrogram(newmat.squeeze(),
                           dt=(final_times[2] - final_times[1]),
                           t0=final_times[0], f0=this_specgram.f0,
                           df=this_specgram.df, unit=this_specgram.unit,
                           name=this_specgram.name + other_specgram.name,
                           channel=None)

    @property
    def std_spectrum(self):
        return MagSpectrum.from_freqseries(self.spectrogram.std(axis=0))


class SchumannPeak(models.Lorentz1D):
    """docstring for MagModel"""


class MagSpectrum(FrequencySeries):
    """
    magnetic spectrum
    """
    @classmethod
    def from_freqseries(cls, f):
        return cls(f)

    def fit(self, peaks, q_factors, amplitudes,
            fitter=fitting.LevMarLSQFitter(), **kwargs):
        # start model
        first = True
        schumann_model = None
        for p, q, a in zip(peaks, q_factors, amplitudes):
            if first:
                schumann_model = models.Lorentz1D(a, p, q)
                first = False
            else:
                schumann_model += models.Lorentz1D(a, p, q)
        # schumann_model += models.Linear1D(-1e-4, 1e-3)
        fit = fitter(schumann_model, self.frequencies.value,
                     self.value, **kwargs)
        self.lorentzian_parameters = fit.parameters
        return fit, schumann_model

    def coarse_grain(self, flow, df):
        return type(self).from_freqseries(coarseGrain(self, flow, df))

    def plot_fit(self, fit, *args, **kwargs):
        myp = self.plot(*args, **kwargs)
        ax = myp.gca()
        ax.plot(self.frequencies.value,
                fit(self.frequencies.value))
        return myp


class MagTimeSeries(TimeSeries):

    @classmethod
    def from_coughlin_matfile(cls, matfile, which='1', calibration=1./16563, sample_rate=128,
                              unit=units.nT, name=None):
        from scipy.io import loadmat
        from gwpy.time import tconvert

        mymat = loadmat(matfile, squeeze_me=True)
        ts = mymat['tt']
        day = tconvert(date.fromordinal(int(np.floor(ts[0])) - 366))
        if which=='1':
            return cls(mymat['data1']*calibration, sample_rate=sample_rate, t0=day, unit=unit, name=name)
        if which=='2':
            return cls(mymat['data2']*calibration, sample_rate=sample_rate, t0=day, unit=unit, name=name)

    @classmethod
    def from_timeseries(cls, ts):
        return cls(ts)

    def magasd(self, *args, **kwargs):
        return MagSpectrum.from_freqseries(self.asd(*args, **kwargs))

    def magfft(self, *args, **kwargs):
        return MagSpectrum.from_freqseries((self.fft(*args, **kwargs)))

    def magpsd(self, *args, **kwargs):
        return MagSpectrum.from_freqseries((self.psd(*args, **kwargs)))

    def magcsd(self, *args, **kwargs):
        return MagSpectrum.from_freqseries((self.csd(*args, **kwargs)))

    @classmethod
    def fetch_lemi(cls, ifo, direction, st, et):
        # check direction
        if direction.upper() != 'X' and direction.upper() != 'Y':
            raise ValueError('Direction must be X or Y')

        # set channels for LEMIs
        if ifo == 'L1':
            chan = ('L1:PEM-EY_VAULT_MAG_LEMI_%s_DQ'
                    % direction.upper())
        elif ifo == 'H1':
            chan = ('H1:PEM-VAULT_MAG_1030X195Y_COIL_%s_DQ'
                    % direction.upper())
        else:
            raise ValueError('ifo must be L1 or H1')
        print(cls)
        # fetch, set units, multiply by calibration
        data = TimeSeries.fetch(chan, st, et) * 0.305
        data.override_unit(units.pT)
        return cls.from_timeseries(data)

    @classmethod
    def find_lemi(cls, ifo, direction, *args, **kwargs):
        # check direction
        if direction.upper() != 'X' and direction.upper() != 'Y':
            raise ValueError('Direction must be X or Y')

        # set channels for LEMIs
        if ifo == 'L1':
            chan = ('L1:PEM-EY_VAULT_MAG_LEMI_%s_DQ'
                    % direction.upper())
        elif ifo == 'H1':
            chan = ('H1:PEM-VAULT_MAG_1030X195Y_COIL_%s_DQ'
                    % direction.upper())
        else:
            raise ValueError('ifo must be L1 or H1')

        # supply the correct frame type no matter what.
        kwargs['frametype'] = '%s_R' % ifo

        # fetch, set units, multiply by calibration.
        return cls.find(chan, *args, **kwargs).override_unit(units.pT) * 0.305


def fit_params_to_dict(fit_params, st):
    from collections import OrderedDict
    peakstr = 'peak%d'
    ampstr = 'amp%d'
    qstr = 'Q%d'
    npeaks = int(np.size(fit_params) / 3)
    newdict = OrderedDict()
    for ii in range(npeaks):
        newdict[peakstr % ii] = fit_params[ii * 3 + 1]
        newdict[ampstr % ii] = fit_params[ii * 3 + 0]
        newdict[qstr % ii] = fit_params[ii * 3 + 2]
    newdict['starttime'] = st
    return newdict


class SchumannParamTable(QTable):
    """
    parameter table for schumann"""

    @classmethod
    def from_fit(cls, fit_params, st):
        """
        fit_params: list of fit parameters
        """
        dct = fit_params_to_dict(fit_params, st)
        names = [key for key in dct.keys()]
        dtypes = [float for key in dct.keys()]
        newtab = cls(names=names, dtype=dtypes)
        newtab.add_row(dct)
        return newtab

    def add_fit(self, fit_params, st):
        self.add_row(fit_params_to_dict(fit_params, st))
