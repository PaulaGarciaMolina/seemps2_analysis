from dataclasses import dataclass
from typing import List, Optional, Callable, Union
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PlotParameters:
    xlabel: str = "x"
    ylabel: str = "y"
    zlabel: str = "z"
    xscale: str = "linear"
    yscale: str = "linear"
    title: Optional[str] = None
    legend_title: Optional[str] = None
    legend_labels: Optional[Union[str, List[str]]] = None
    style: Optional[Union[str, List[str]]] = "-o"  # Includes marker, color and linestyle
    save_path: Optional[str] = None


def plot_line(
    x: np.ndarray,
    y: Union[np.ndarray, List[np.ndarray]],
    hold: bool = True,
    axs: plt.Axes = None,
    parameters: PlotParameters = PlotParameters(),
):
    """Plots a line given by a 1d-array (or a collection of them) y in a domain given by x.
    Sets the plot parameters by means of a PlotParameter object and a kwargs dictionary.
    """

    # Create a new figure and axis if axs is not provided
    if axs is None:
        _, axs = plt.subplots()

    # Function to apply mask and plot the line
    def plot_row(x, y, style):
        mask = np.isfinite(y)  # Create a mask to ignore NaN and None values
        axs.plot(np.array(x)[mask], y[mask], style)

    # Plot a single line if y is a 1D array
    if isinstance(y, np.ndarray) and len(y.shape) == 1:  # 1D array
        plot_row(x, y, parameters.style)
    elif isinstance(y, list) or len(y.shape) > 1:  # 2D array of list of arrays
        for idx, col in enumerate(y.T):
            style = parameters.style if isinstance(parameters.style, str) else parameters.style[idx]
            plot_row(x, col, style)

    if parameters.legend_title:
        axs.legend(title=parameters.legend_title, labels=parameters.legend_labels)

    # Set plot parameters
    axs.set_title(parameters.title)
    axs.set_xlabel(parameters.xlabel)
    axs.set_ylabel(parameters.ylabel)
    axs.set_title(parameters.title)
    axs.set_xscale(parameters.xscale)
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
    X: np.ndarray,
    Y: np.ndarray,
    Z: Union[np.ndarray, List[np.ndarray]],
    hold: bool = False,
    axs: plt.Axes = None,
    parameters: PlotParameters = PlotParameters(),
    **kwargs,
):
    """Plots a surface given by 2d-array (or a collection of them) y in a domain given by x.
    Sets the plot parameters by means of a PlotParameter object and a kwargs dictionary.
    """
    if axs is None:
        _, axs = plt.subplots(subplot_kw={"projection": "3d"})

    def plot_matrix(X, Y, Z):
        axs.plot_surface(X, Y, Z)

    # Plot a surface if Z is a 2D array
    if isinstance(Z, np.ndarray) and len(Z.shape) == 2:  # 2D array
        plot_matrix(X, Y, Z)
    elif isinstance(Z, list):  # List of 2D arrays
        for idx, matrix in enumerate(Z):
            # style = parameters.style if isinstance(parameters.style, str) else parameters.style[idx]
            plot_matrix(X, Y, matrix)
    else:
        raise ValueError("Invalid array")

    if parameters.legend_title:
        axs.legend(title=parameters.legend_title, labels=parameters.legend_labels)

    # Set plot parameters
    axs.set_title(parameters.title)
    axs.set_xlabel(parameters.xlabel)
    axs.set_ylabel(parameters.ylabel)
    axs.set_zlabel(parameters.zlabel)
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


def plot_contour(
    X: np.ndarray,
    Y: np.ndarray,
    Z: Union[np.ndarray, List[np.ndarray]],
    hold: bool = False,
    axs: plt.Axes = None,
    parameters: PlotParameters = PlotParameters(),
    **kwargs,
):
    """Plots a contour given by a 2d-array (or a collection of them) y in a domain given by x.
    Sets the plot parameters by means of a PlotParameter object and a kwargs dictionary.
    """
    if axs is None:
        _, axs = plt.subplots()

    def plot_matrix(X, Y, Z, style):
        CS = axs.contour(X, Y, Z, levels=10)
        axs.clabel(CS, inline=True, fontsize=10)

    # Plot a surface if Z is a 2D array
    if isinstance(Z, np.ndarray) and len(Z.shape) == 2:  # 2D array
        plot_matrix(X, Y, Z, parameters.style)
    elif isinstance(Z, list):  # List of 2D arrays
        for idx, matrix in enumerate(Z):
            style = parameters.style if isinstance(parameters.style, str) else parameters.style[idx]
            plot_matrix(X, Y, matrix, style)
    else:
        raise ValueError("Invalid array")

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


def set_mosaic(
    rows: int = 1,
    cols: int = 2,
    figsize: tuple = (12, 10),
) -> plt.Axes:
    """Returns a plt.Axes object in order to insert figures in a subplot of given columns and rows."""
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    return fig, axs
