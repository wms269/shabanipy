# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2019 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Carriers density analysis.

"""
import numpy as np
import scipy.constants as cs
import matplotlib.pyplot as plt
from lmfit.models import LinearModel


def extract_density(field, rxy, field_cutoffs, plot_fit=False):
    """Extract the carriers density from the low field resistance dependence.

    The extraction relies on a simple linear fit performed on a specified field
    range.

    Parameters
    ----------
    field : np.ndarray
        Magnetic field values for which the the transverse resistance was
        measured.
        This can be a multidimensional array in which case the last
        dimension will be considered as the swept dimension.
    rxy : np.ndarray
        Transverse resistance values which were measured.
        This can be a multidimensional array in which case the last dimension
        will be considered as the swept dimension.
    field_cutoffs : tuple | np.ndarray
        Pairs of low/high field values on which to perform the linear fit. If
        only one pair of value is provided it will be used for all fits.

    Returns
    -------
    densities : float | np.ndarray
        Densities extracted from the linear fit. Will be a float if a single
        value was extracted from the provided data.

    densities_stderr : float | np.ndarray
        Incertitude on the density expressed as the standard deviation.
        Will be a float if a single value was extracted from the provided data.

    """
    # Identify the shape of the data and make them suitable for the following
    # treatment.
    if len(field.shape) >= 2:
        input_is_1d = False
        original_shape = field.shape[:-1]
        trace_number = np.prod(original_shape)
        field = field.reshape((trace_number, -1))
        rxy = rxy.reshape((trace_number, -1))
        if len(field_cutoffs) == 2:
            fc = np.empty(original_shape + (2,))
            fc[..., 0] = field_cutoffs[0]
            fc[..., 1] = field_cutoffs[1]
            field_cutoffs = fc
        field_cutoffs = field_cutoffs.reshape((trace_number, -1))
    else:
        input_is_1d = True
        trace_number = 1
        field = np.array((field,))
        rxy = np.array((rxy,))
        field_cutoffs = np.array((field_cutoffs,))

    results = np.empty((2, trace_number))
    model = LinearModel()

    # Perform a linear fit in the specified field range and extract the slope
    for i in range(trace_number):
        start_field, stop_field = field_cutoffs[i]
        start_ind = np.argmin(np.abs(field[i] - start_field))
        stop_ind = np.argmin(np.abs(field[i] - stop_field))
        start_ind, stop_ind =\
            min(start_ind, stop_ind), max(start_ind, stop_ind)
        f = field[i][start_ind:stop_ind]
        r = rxy[i][start_ind:stop_ind]
        res = model.fit(r, x=f)
        results[0, i] = 1/res.best_values['slope']/cs.e  # value in m^-2
        results[1, i] = results[0, i] * (res.params['slope'].stderr /
                                         res.best_values['slope'])

        # If requested plot the result in a dedicated window.
        if plot_fit:
            plt.figure()
            plt.plot(f, r, '+')
            plt.plot(f, res.best_fit)
            plt.xlabel('Field')
            plt.ylabel('Rxy')
            plt.tight_layout()

    if input_is_1d:
        return results[:, 0]
    else:
        return results.reshape((2, ) + original_shape)
