"""Utility functions to analyse V-I characteristic.

"""
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LinearModel

from .utils import compute_voltage_offset, extract_switching_current


def analyse_vi_curve(
    current_bias: np.ndarray,
    measured_voltage: np.ndarray,
    voltage_offset_correction: Union[int, float],
    ic_voltage_threshold: float,
    high_bias_threshold: float,
    plots: bool = True,
    plot_title: str = "",
):
    """Extract the critical and excess current along with the normal resistance.

    All values are extracted for cold and hot electrons. The cold side is the
    one on which the bias is ramped from 0 to large value, the hot one the one
    on which bias is ramped towards 0.

    If more than 1D array input are used, the last dimension if assumed to be swept.
    The current sweep does not have to be the same for all outer dimensions.

    Parameters
    ----------
    current_bias : np.ndarray
        Current bias applied on the junction in A.
    measured_voltage : np.ndarray
        Voltage accross the junction in V.
    voltage_offset_correction : int | float
        Number of points around 0 bias on which to average to correct for the
        offset in the DC measurement, or actual offset to substract.
    ic_voltage_threshold : float
        Voltage threshold in V above which the junction is not considered to carry a
        supercurrent anymore. Used in the determination of the critical current.
    high_bias_threshold : float
        Positive bias value above which the data can be used to extract the
        normal resistance.
    plots : bool, optional
        Generate summary plots of the fitting.

    Returns
    -------
    offset : np.ndarray
        Offset used to correct the measured voltage.
    rn_c : np.ndarray
        Normal resistance evaluated on the cold electron side.
    rn_h : np.ndarray
        Normal resistance evaluated on the hot electron side.
    ic_c : np.ndarray
        Critical current evaluated on the cold electron side.
    ic_h : np.ndarray
        Critical current evaluated on the hot electron side.
    iexe_c : np.ndarray
        Excess current evaluated on the cold electron side.
    iexe_h : np.ndarray
        Excess current evaluated on the hot electron side.

    """
    # Allocate output arrays
    first_bias = current_bias[..., 0]
    offset = np.empty_like(first_bias)
    rn_cold = np.empty_like(first_bias)
    rn_hot = np.empty_like(first_bias)
    ic_cold = np.empty_like(first_bias)
    ic_hot = np.empty_like(first_bias)
    iexe_cold = np.empty_like(first_bias)
    iexe_hot = np.empty_like(first_bias)

    it = np.nditer(current_bias[..., 0], ["multi_index"])
    for first_bias in it:

        # Get the 1D sweep (since we need it for fitting)
        bias = current_bias[it.multi_index]
        volt = measured_voltage[it.multi_index]

        # Determine the hot and cold electron side
        if first_bias < 0.0:
            cold_value = lambda p, n: abs(p)
            hot_value = lambda p, n: abs(n)
        else:
            cold_value = lambda p, n: abs(n)
            hot_value = lambda p, n: abs(p)
            # Also flipped the data so that we always go from negative
            # to positive values
            bias = bias[::-1]
            volt = volt[::-1]

        # Index at which the bias current is zero
        index = np.argmin(np.abs(bias))

        # Correct the offset in the voltage data
        if voltage_offset_correction and isinstance(voltage_offset_correction, float):
            v_offset = voltage_offset_correction
        elif voltage_offset_correction and isinstance(voltage_offset_correction, int):
            v_offset = compute_voltage_offset(bias, volt, voltage_offset_correction)

        volt -= v_offset

        # Extract the critical current on the positive and negative branch
        ic_p = extract_switching_current(bias, volt, ic_voltage_threshold, "positive")
        ic_n = extract_switching_current(bias, volt, ic_voltage_threshold, "negative")

        # Fit the high positive/negative bias to extract the normal resistance
        # excess current and their product
        index_pos = np.argmin(np.abs(bias - high_bias_threshold))
        index_neg = np.argmin(np.abs(bias + high_bias_threshold))

        model = LinearModel()
        pars = model.guess(volt[index_pos:], x=bias[index_pos:])
        pos_results = model.fit(volt[index_pos:], pars, x=bias[index_pos:])

        pars = model.guess(volt[index_neg:], x=bias[index_neg:])
        neg_results = model.fit(volt[:index_neg], pars, x=bias[:index_neg])

        rn_p = pos_results.best_values["slope"]
        # Iexe p
        iexe_p = -pos_results.best_values["intercept"] / rn_p

        rn_n = neg_results.best_values["slope"]
        # Iexe n
        iexe_n = neg_results.best_values["intercept"] / rn_n

        offset[it.multi_index] = v_offset
        for arrays, values in zip(
            [(rn_cold, rn_hot), (ic_cold, ic_hot), (iexe_cold, iexe_hot)],
            [(rn_p, rn_n), (ic_p, ic_n), (iexe_p, iexe_n),],
        ):
            arrays[0][it.multi_index] = cold_value(*values)
            arrays[1][it.multi_index] = hot_value(*values)

        if plots:
            # Prepare a summary plot: full scale
            fig = plt.figure(constrained_layout=True)
            fig.suptitle(plot_title)
            ax = fig.gca()
            ax.plot(bias * 1e6, volt * 1e3)
            ax.plot(
                bias[index:] * 1e6,
                model.eval(pos_results.params, x=bias[index:]) * 1e3,
                "--k",
            )
            ax.plot(
                bias[: index + 1] * 1e6,
                model.eval(neg_results.params, x=bias[: index + 1]) * 1e3,
                "--k",
            )
            ax.set_xlabel("Bias current (µA)")
            ax.set_ylabel("Voltage drop (mV)")

            # Prepare a summary plot: zoomed in
            mask = np.logical_and(np.greater(bias, -3 * ic_p), np.less(bias, 3 * ic_p))
            if np.any(mask):
                fig = plt.figure(constrained_layout=True)
                fig.suptitle(plot_title + ": zoom")
                ax = fig.gca()
                ax.plot(bias * 1e6, volt * 1e3)
                aux = model.eval(pos_results.params, x=bias[index:]) * 1e3
                ax.plot(
                    bias[index:] * 1e6,
                    model.eval(pos_results.params, x=bias[index:]) * 1e3,
                    "--",
                )
                ax.plot(
                    bias[: index + 1] * 1e6,
                    model.eval(neg_results.params, x=bias[: index + 1]) * 1e3,
                    "--",
                )
                ax.set_xlim(
                    (
                        -3 * cold_value(ic_p, ic_n) * 1e6,
                        3 * cold_value(ic_p, ic_n) * 1e6,
                    )
                )
                aux = volt[mask]
                ax.set_ylim((np.min(aux[mask]) * 1e3, np.max(aux[mask]) * 1e3,))
                ax.set_xlabel("Bias current (µA)")
                ax.set_ylabel("Voltage drop (mV)")

    if current_bias.ndim == 1:
        offset = float(offset)
        rn_cold = float(rn_cold)
        rn_hot = float(rn_hot)
        ic_cold = float(ic_cold)
        ic_hot = float(ic_hot)
        iexe_cold = float(iexe_cold)
        iexe_hot = float(iexe_hot)

    return (
        offset,
        rn_cold,
        rn_hot,
        ic_cold,
        ic_hot,
        iexe_cold,
        iexe_hot,
    )
