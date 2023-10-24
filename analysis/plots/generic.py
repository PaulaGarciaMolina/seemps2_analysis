from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PlotParameters:
    title: str = "Figure"
    xlabel: str = "x"
    ylabel: str = "y"
    zlabel: str = "z"
    legend: List = None
    legend_title: str = "Legend"
    marker: str = "-o"
    logscale: bool = False
    save_path: str = None


def plot_1d_array(
    x: np.ndarray,
    y_1: np.ndarray,
    y_2: Optional[np.ndarray] = None,
    hold: bool = False,
    axs: plt.Axes = plt.figure(),
    parameters: PlotParameters = PlotParameters(),
    **kwargs,
):
    """Plots a 1d array y_1 in a domain given by x. Optionally plots another array y_2
    in the same domain, and allows for plotting more arrays by means of the hold parameter.
    Sets the plot parameters by means of a PlotParameter object and a kwargs dictionary.
    """
    pass


def plot_2d_array(
    x_1: np.ndarray,
    x_2: np.ndarray,
    y_1: np.ndarray,
    y_2: Optional[np.ndarray] = None,
    hold: bool = False,
    axs: plt.Axes = None,
    parameters: PlotParameters = PlotParameters(),
    **kwargs,
):
    """Plots a 2d array y_1 in a domain given by the meshgrid of x_1 and x_2. Optionally, plots
    another 2d array y_2 in the same domain, and allows for plotting more arrays by means of
    the hold parameter."""
    pass


def plot_mosaic_2(
    plot_function_1: Callable, plot_function_2: Callable, suptitle: str = "Figure"
):
    """Plots two figures given respectively by two functions by passing them an axis object."""
    pass
