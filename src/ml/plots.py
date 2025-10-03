"""Plotting functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from numpy import ndarray
import pandas as pd

from ml.logs import logger
from ml.utils import Utils


class Plotter:
    """Plotting functions."""

    @staticmethod
    def plot_forecast(
        actual: ndarray,
        predicted: ndarray,
        results_subdir: str,
        title: str,
        figsize: tuple[int, int] = (20, 3),
        **kwargs,
    ):
        """Plot forecasted vs actual values

        :param ndarray actual: Original time series values
        :param ndarray predicted: Forecasted values
        :param str results_subdir: Path to output directory
        :param str title: Title (library or model name)
        """
        plt.clf()
        pd.plotting.register_matplotlib_converters()
        # Create plot
        plt.figure(0, figsize=figsize)  # Pass plot ID to prevent memory issues
        plt.plot(actual, label='actual')
        plt.plot(predicted, label='predicted')
        save_path = Path(results_subdir, 'plots', f'{title}.svg')
        Path(results_subdir, 'plots').mkdir(parents=True, exist_ok=True)
        logger.debug(f'Saving plot: {save_path}')
        Utils.save_plot(title, save_path=save_path, **kwargs)

    @staticmethod
    def save_plot(
        title,
        xlabel=None,
        ylabel=None,
        suptitle='',
        show=False,
        legend=None,
        save_path=None,
        yscale='linear',
        **kwargs,
    ):
        """Apply title and axis labels to plot. Show and save to file. Clear plot.

        :param title: Title for plot
        :param xlabel: Plot X-axis label
        :param ylabel: Plot Y-axis label
        :param title: Subtitle for plot
        :param show: Show plot on screen, defaults to False
        :param legend: Legend, defaults to None
        :param save_path: Save plot to file if not None, defaults to None
        :param yscale: Y-Scale ('linear' or 'log'), defaults to 'linear'
        """
        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.yscale(yscale)

        plt.title(title)
        plt.suptitle(suptitle)
        if legend is not None:
            plt.legend(legend, loc='upper left')

        # Show plot
        if show:
            plt.show()
        # Show plot as file
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        # Clear for next plot
        plt.cla()
        plt.clf()
        plt.close('all')
