# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright 2020 by ShabaniPy Authors, see AUTHORS for more details.
#
# Distributed under the terms of the MIT license.
#
# The full license is in the file LICENCE, distributed with this software.
# -----------------------------------------------------------------------------
"""Utility functions to analyse JJ data.

"""
from typing import Union

import numpy as np
from typing_extensions import Literal


def compute_voltage_offset(
    current_bias: np.ndarray,
    measured_voltage: np.ndarray,
    voltage_offset_correction: int,
) -> Union[float, np.ndarray]:
    """Compute the voltage offset in the VI characteristic of a JJ.

    If more than 1D array input are used, the last dimension if assumed to be swept
    and the 0 of the bias is assumed to be the same for all sweeps.

    Parameters
    ----------
    current_bias : np.ndarray
        Current bias applied on the junction.
    measured_voltage : np.ndarray
        Voltage accross the junction.
    voltage_offset_correction : int
        Number of points around 0 bias on which to average to correct for the
        offset in the DC measurement.

    Returns
    -------
    offset : Union
        Offset on the measured voltage.

    """
    # Index at which the bias current is zero
    # We access the first element of each axis except the last for which we get
    # the complete sweep.
    index = np.argmin(
        np.abs(
            current_bias[tuple([0 for i in current_bias.shape[:-1]] + [slice(None)])]
        )
    )

    # Correct the offset in the voltage data and make sure we only use positive
    # indexes in slices
    start = index - voltage_offset_correction + 1
    avg_sl = slice(start if start >= 0 else 0, index + voltage_offset_correction)
    to_average = measured_voltage[..., avg_sl]

    return np.average(to_average, axis=-1)


def extract_switching_current(
    bias: np.ndarray,
    volt_or_res: np.ndarray,
    threshold: float,
    side: Literal["positive", "negative"] = "positive",
    offset_correction=0,
) -> np.ndarray:
    """Extract the switching current from a voltage or resistance map.

    If more than 1D array input are used, the last dimension if assumed to be swept.
    The current sweep does not have to be the same for all outer dimensions.

    Parameters
    ----------
    bias : np.ndarray
        Bias current applied to the junction.
    diff : np.ndarray
        Differential resistance of the junction.
    threshold : float
        Threshold value used to determine the critical current.
    side : {"positive", "negative"}, optional
        On which branch of the bias current to extract the critical current,
        by default "positive"
    offset_correction : int, optional
        Number of points to use to compute an offset correction of the data
        around 0 bias, by default 0

    Returns
    -------
    np.ndarray
        Extracted critical current.

    """
    if side not in ("positive", "negative"):
        raise ValueError(f"Side should be 'positive' or 'negative', found {side}.")

    if offset_correction:
        offset = compute_voltage_offset(bias, volt_or_res, offset_correction)
    else:
        offset = 0

    # Index at which the bias current is zero
    mask = np.greater_equal(bias, 0) if side == "positive" else np.less_equal(bias, 0)
    if not mask.any():
        raise ValueError(f"No {side} bias data in the set.")

    # Mask the data to get only the data we care about
    masked_bias = bias[mask].reshape(bias.shape[:-1] + (-1,))
    masked_data = volt_or_res[mask].reshape(bias.shape[:-1] + (-1,))
    it = np.nditer(masked_bias[..., 0], ["multi_index"])
    for b in it:
        if np.argmin(np.abs(masked_bias[it.multi_index])) != 0:
            masked_bias[it.multi_index + (slice(None, None),)] = masked_bias[
                it.multi_index + (slice(None, None, -1),)
            ]
            masked_data[it.multi_index + (slice(None, None),)] = masked_data[
                it.multi_index + (slice(None, None, -1),)
            ]

    # Axis manipulation since we have (n, m, l) and (n, m) and we can only
    # broadcast (l, m, n) and (m, n)
    temp = np.moveaxis(np.moveaxis(masked_data, -1, 0) - offset, 0, -1)
    temp = np.greater(np.abs(temp), threshold)
    # Make sure we pinpoint the last current where we were below threshold
    index = np.argmax(temp, axis=-1) - 1

    return np.take_along_axis(masked_bias, index[..., None], axis=-1).reshape(
        bias.shape[:-1]
    )
