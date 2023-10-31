from dataclasses import dataclass
from typing import List, Optional, Callable, Union
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PlotParameters:
    xlabel: str = "x"
    ylabel: str = "y"
    zlabel: str = "z"
    yscale: str = "linear"
    legend_title: Optional[str] = None
    title: Optional[str] = None
    legend_labels: Optional[Union[str, List[str]]] = None
    marker: Optional[Union[str, List[str]]] = "-o"
    color: Optional[Union[str, List[str]]] = None
    save_path: Optional[str] = None


def plot_line(
    x: np.ndarray,
    y: Union[np.ndarray, List[np.ndarray]],
    hold: bool = True,
    axs: plt.Axes = None,
    parameters: PlotParameters = PlotParameters(),
    **kwargs,
):
    """Plots a line given by a 1d-array (or a collection of them) y in a domain given by x.
    Sets the plot parameters by means of a PlotParameter object and a kwargs dictionary.
    """

    # Create a new figure and axis if axs is not provided
    if axs is None:
        _, axs = plt.subplots()

    # Function to apply mask and plot the line
    def plot_single_line(x, y, parameters, **kwargs):
        mask = np.isfinite(y)  # Create a mask to ignore NaN and None values
        axs.plot(np.array(x)[mask], y[mask], parameters.marker, color=parameters.color, **kwargs)

    # Plot a single line if y is a 1D array
    if isinstance(y, np.ndarray):
        for i in range(y.shape[1]):
            plot_single_line(x, y[:, i], parameters, **kwargs)
    elif isinstance(y, list):
        for line in y:
            if isinstance(line, np.ndarray):
                plot_single_line(x, line, parameters, **kwargs)

    if parameters.legend_title:
        axs.legend(title=parameters.legend_title, labels=parameters.legend_labels)

    # Set plot parameters
    axs.set_title(parameters.title)
    axs.set_xlabel(parameters.xlabel)
    axs.set_ylabel(parameters.ylabel)
    axs.set_title(parameters.title)
    axs.set_yscale(parameters.yscale)

    # Save the figure
    if parameters.save_path:
        plt.savefig(parameters.save_path)

    # Show or hold the plot as specified
    if hold:
        plt.show(block=False)
    else:
        plt.show()


def plot_surface(
    x: np.ndarray,
    y: Union[np.ndarray, List[np.ndarray]],
    hold: bool = False,
    axs: plt.Axes = None,
    parameters: PlotParameters = PlotParameters(),
    **kwargs,
):
    """Plots a surface given by 2d-array (or a collection of them) y in a domain given by x.
    Sets the plot parameters by means of a PlotParameter object and a kwargs dictionary.
    """
    pass


def plot_contour(
    x: np.ndarray,
    y: Union[np.ndarray, List[np.ndarray]],
    hold: bool = False,
    axs: plt.Axes = None,
    parameters: PlotParameters = PlotParameters(),
    **kwargs,
):
    """Plots a contour given by a 2d-array (or a collection of them) y in a domain given by x.
    Sets the plot parameters by means of a PlotParameter object and a kwargs dictionary.
    """
    pass


def set_mosaic(
    rows: int = 1,
    cols: int = 2,
    figsize: tuple = (12, 10),
) -> plt.Axes:
    """Returns a plt.Axes object in order to insert figures in a subplot of given columns and rows."""
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    return fig, axs
