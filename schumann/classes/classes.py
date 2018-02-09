from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.spectrogram import Spectrogram
from .coarseGrain import coarseGrain
from astropy import units
from astropy.modeling import models, fitting
from h5py import File as h5File
from datetime import date
from astropy.time import Time
from astropy.table import QTable
import os
import numpy as np

__all__ = ['CoughlinMagFile',
           'SchumannPeak',
           'MagSpectrum',
           'MagTimeSeries',
           'SchumannParamTable']


class CoughlinMagFile(h5File):
    """
    Info from a "coughlin data file"
    """
    @property
    def start_time(self):
        # date string
        fname = os.path.basename(self.filename).split('.')[0]
        datestr = int(fname.split('-')[0])
        # date object
        mydate = date.fromordinal(datestr - 366)
        # use astropy to get gps time from datestring
        return Time(mydate.strftime('%Y-%m-%d')).gps

    @property
    def segdur(self):
        # segment duration
        return self['data']['T'].value.squeeze()

    @property
    def frequencies(self):
        # list of frequencies from file
        return self['data']['ff'].value.squeeze()

    @property
    def df(self):
        return self.frequencies[2] - self.frequencies[1]

    @property
    def f0(self):
        return self.frequencies[0]

    @property
    def spectrogram(self):
        return Spectrogram(self['data']['spectra'].value.squeeze(),
                           df=self.df, f0=self.df,
                           t0=self.start_time, dt=self.segdur,
                           name='magnetic data',
                           channel=None) * units.nT

    @property
    def average_spectrum(self):
        return MagSpectrum.from_freqseries(self.spectrogram.mean(axis=0))

    def average_spectrum_cropped(self, st, et):
        return MagSpectrum.from_freqseries(self.spectrogram.crop(st, et).mean(axis=0))

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
        return cls(f.value, df=f.df,
                   unit=f.unit, f0=f.f0,
                   name=f.name, channel=f.channel
                   )

    def fit(self, peaks, q_factors, amplitudes,
            fitter=fitting.LevMarLSQFitter(), **kwargs):
        # start model
        First = True
        for p, q, a in zip(peaks, q_factors, amplitudes):
            if First:
                schumann_model = models.Lorentz1D(a, p, q)
                First = False
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
    def fetch_lemi(cls, ifo, direction, *args, **kwargs):
        # check direction
        if direction.upper() != 'X' or direction.upper() != 'Y':
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

        # fetch, set units, multiply by calibration
        return cls.fetch(chan, *args, **kwargs).override_unit(units.pT) * 0.305

    @classmethod
    def find_lemi(cls, ifo, direction, *args, **kwargs):
        # check direction
        if direction.upper() != 'X' or direction.upper() != 'Y':
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
