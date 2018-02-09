from __future__ import division
from math import floor
from numpy import array, append
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from timeit import timeit


def coarseGrain(spectrum, flowy=None, deltaFy=None, Ny=None):
    """
    Coarse grain a frequency-series. Returns a :py:class:`gwpy.frequencyseries`
    structure coarse-grained to the frequency values f = flowy + deltaFy * [0:Ny-1]

    Parameters
    ----------
    flowy : `float`
        Starting frequency value for the coarse grained
        spectrum. Called f0 in GWPy FrequencySeries. The
        default value is the minimum allowed.
    deltaFy : `float`
        Step size of the coarse grained spectrum. The 
        default is the minimum allowed.
    Ny : `int`
        Number of steps. The default is the maximum amount
        with the given step size and starting frequency.
    
    Original routine written in MATLAB by Joseph D. Romano.
    """

    # Extract metadata
    data = spectrum.value
    Nx = len(data)
    flowx = spectrum.f0.value
    deltaFx = spectrum.df.value

    # Calculate default values
    if deltaFy == None:
        deltaFy = deltaFx

    if flowy == None:
        flowy = flowx + (deltaFy - deltaFx) * 0.5
    
    def calc_max(Nx, flowx, deltaFx, flowy, deltaFy):
        fhighx = flowx + (Nx - 1) * deltaFx
        Ny = ((fhighx + 0.5 * (deltaFx - deltaFy) - flowy) / deltaFy) + 1
        return int(Ny)

    if Ny == None:
        Ny = calc_max(Nx, flowx, deltaFx, flowy, deltaFy)

    ##########################
    # Error Checking (start) #
    ##########################

    # Check that frequency spacings and number of elements are > 0.
    if (deltaFx or deltaFy or Ny) < 0:
        raise Exception('Bad input arguments')

    # Check that freq resolution of x is finer than the desired resolution.
    if deltaFy < deltaFx:
        raise Exception('deltaF coarse-grain < deltaF fine-grain')

    # Check desired start frequency for coarse-grained series.
    if (flowy - 0.5 * deltaFy) < (flowx - 0.5 * deltaFx):
        raise Exception('Desired coarse-grained start frequency is too low')

    # Check desired stop frequency for coarse-grained series.
    fhighx = flowx + (Nx - 1) * deltaFx
    fhighy = flowy + (Ny - 1) * deltaFy
    if (fhighy + 0.5 * deltaFy) > (fhighx + 0.5 * deltaFx):
        raise Exception('Desired coarse-grained stop frequency is too high')

    ########################
    # Error Checking (end) #
    ########################

    # Indices for coarse-grained series.
    i = array([num for num in range(Ny)])

    # Define modified floor function to account for precision error.
    def floor1(index):
        if abs(index - round(index)) < 1e-8:
            indexC = int(round(index))
        else:
            indexC = int(floor(index))
        return indexC

    # Calculate the low and high indices of fine-grained series.
    jlow = array([1 + floor1((flowy + deltaFy * (i - 0.5)[k] - flowx -
                              0.5 * deltaFx) / deltaFx)
                  for k in range(len(i))])
    jhigh = array([1 + floor1((flowy + deltaFy * (i + 0.5)[k] - flowx -
                               0.5 * deltaFx) / deltaFx)
                  for k in range(len(i))])

    # Calculate fractional contributions.
    fraclow = array([(flowx + deltaFx * (jlow + 0.5)[k] - flowy -
                      deltaFy * (i - 0.5)[k]) / deltaFx
                     for k in range(len(i))])
    frachigh = array([(flowy - deltaFx * (jhigh - 0.5)[k] -
                       flowx + deltaFy * (i + 0.5)[k]) / deltaFx
                     for k in range(len(i))])

    ###################################
    # Calculate coarse-grained values #
    ###################################

    # Sum of middle terms.
    def sumTerms(x, jlow, jhigh):
        Ny = len(jlow)
        return array([sum(x[(jlow[k] - 1):jhigh[k]]) for k in range(Ny)])
    
    jtemp = array([num + 2 for num in jlow])
    midsum = sumTerms(data, jtemp, jhigh)

    # Calculate all but final value of the new coarse spectrum
    ya = array([(data[jlow[k]] * fraclow[k] + data[jhigh[k]] *
                 frachigh[k] + midsum[k]) * (deltaFx / deltaFy)
               for k in range(Ny - 1)])

    # Calculate final value of the coarse spectrum (this is a scalar).
    if jhigh[-1] > (Nx - 1):
        # Special case when jhigh exceeds maximum allowed value.
        yb = (deltaFx / deltaFy) * (data[jlow[-1]] * fraclow[-1] +
                                    midsum[-1])
    else:
        yb = (deltaFx / deltaFy) * (data[jlow[-1]] * fraclow[-1] +
                                    data[jhigh[-1]] * frachigh[-1] +
                                    midsum[-1])

    # Fill structure for coarsed-grained frequency series.
    coarseSpectrum = append(ya, yb)
    newFrequencySeries = FrequencySeries(coarseSpectrum, f0=flowy, df=deltaFy,
                                         channel=spectrum.channel,
                                         epoch=spectrum.epoch,
                                         unit=spectrum.unit)
    return newFrequencySeries

