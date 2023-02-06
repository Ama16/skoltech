import numpy as np
from etna.analysis.utils import prepare_axes

from FACI.compute_beta import computeConfInt


def plot_coverage_level(ax, fixed_error, mean_error, adapt_error, alpha=0.1, to_use=150):
    mean_error = [1-np.mean(mean_error[i:i+to_use]) for i in range(len(mean_error)-to_use+1)]
    ax.plot(mean_error, label="mean")
    
    fixed_error = [1-np.mean(fixed_error[i:i+to_use]) for i in range(len(fixed_error)-to_use+1)]
    ax.plot(fixed_error, label="fixed")
    
    adapt_error = [1-np.mean(adapt_error[i:i+to_use]) for i in range(len(adapt_error)-to_use+1)]
    ax.plot(adapt_error, label="bernouli")
    
    ax.legend()
    ax.grid()
    ax.hlines(1-alpha, 0, len(adapt_error), colors="black")


def plot_with_intervals(ts, res_final, allRes, col2use="mae"):
    segments = np.unique(res_final["segment"])
    _, ax = prepare_axes(len(segments), 1, figsize=(16, 10))
    
    for idx, seg in enumerate(segments):
        cint = computeConfInt(res_final[res_final["segment"] == seg], allRes[idx][4], col2use=col2use)
        length = len(cint[0])

        values = ts[:, seg, "target"]
        forecasts = res_final[res_final["segment"] == seg]["forecast"]

        ax[idx].plot(values[-length-100:], label="True")

        ax[idx].plot(values[-len(forecasts[-length:]):].index, forecasts[-length:], label="Forecast")

        ax[idx].fill_between(
            values[-length:].index,
            np.array(cint[0]),
            np.array(cint[1]),
            facecolor='g',
            alpha=0.5,
        )
        ax[idx].legend()
        ax[idx].set_title(seg)