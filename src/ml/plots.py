"""Plotting functions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from numpy import ndarray
import pandas as pd

from ml.logs import logger


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
        Plotter.save_plot(title, save_path=save_path, **kwargs)

    @staticmethod
    def save_plot(title: str, save_path: str | Path | None = None, **kwargs):
        """Apply title and axis labels to plot. Show and save to file. Clear plot.

        :param str title: Title for plot
        :param str | Path | None save_path: Save plot to file if not None, defaults to None
        :param str | None legend: Legend, defaults to None
        :param bool show: Show plot on screen, defaults to False
        :param str suptitle: Subtitle for plot
        :param str xlabel: Plot X-axis label
        :param str ylabel: Plot Y-axis label
        :param str xscale: X-Scale ('linear' or 'log'), defaults to 'linear'
        :param str yscale: Y-Scale ('linear' or 'log'), defaults to 'linear'
        """
        # Axis labels
        if kwargs.get('xlabel'):
            plt.xlabel(kwargs['xlabel'])
        if kwargs.get('ylabel'):
            plt.ylabel(kwargs['ylabel'])
        # Axis scaling
        plt.xscale(kwargs.get('xscale', 'linear'))
        plt.yscale(kwargs.get('yscale', 'linear'))
        # Title and subtitle
        plt.title(title)
        plt.suptitle(kwargs.get('suptitle', ''))
        # Legend
        if kwargs.get('legend'):
            plt.legend(kwargs['legend'], loc='upper left')
        # Show plot
        if kwargs.get('show'):
            plt.show()
        # Save plot as file
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        # Clear for next plot
        plt.cla()
        plt.clf()
        plt.close('all')
