import numpy as np
import scipy.integrate
import scipy.stats
from typing import List


def tv_error(pred, truth) -> np.ndarray:
    if len(pred.shape) == 2:
        diff = np.abs(pred - truth)
        return np.mean(0.5 * diff.sum(axis=-1))
    elif len(pred.shape) == 3:
        diff = np.abs(pred - np.expand_dims(truth, axis=1))
        return np.mean(diff.sum(axis=-1) * 0.5, axis=0)  # length N (one TV error per sample)
    else:
        raise ValueError("Unrecognized pred shape {}".format(pred.shape))


def rms(pred, truth) -> np.ndarray:
    if len(pred.shape) <= 2:
        return np.sqrt(
            np.mean(
                np.square(pred - truth)
            )
        )
    elif len(pred.shape) == 3:
        return np.sqrt(
            np.mean(
                np.square(pred - np.expand_dims(truth, axis=1)),
                axis=(0, 2)
            )
        )
    else:
        raise ValueError("Unrecognized pred shape {}".format(pred.shape))


def compute_rank_corr(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    timeseries_corrs = []
    for pred_t, truth_t in zip(pred, truth):
        # pred_t may be 2-dim or 1-dim
        if len(pred_t.shape) == 1:
            res = scipy.stats.kendalltau(pred_t, truth_t)
            timeseries_corrs.append(res.correlation)
        elif len(pred_t.shape) == 2:
            stats = np.array([
                scipy.stats.kendalltau(pred_t_sample, truth_t).correlation
                for pred_t_sample in pred_t
            ])
            timeseries_corrs.append(stats)
        else:
            raise ValueError("Unrecognized pred shape {}".format(pred.shape))
    timeseries_corrs = np.stack(timeseries_corrs, axis=0)  # shape is (T) or (T,N)
    return np.mean(timeseries_corrs, axis=0)


def strain_split_rms(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """
    Evaluate RMSE separately for each trajectory
    """
    # each array has shape (T, S)
    _, S = pred.shape
    rmse_arr = np.zeros(S, dtype=float)
    for s_idx in range(S):
        traj_pred = pred[:, s_idx]
        traj_truth = truth[:, s_idx]
        rmse_arr[s_idx] = rms(traj_truth, traj_pred)
    return rmse_arr


def binned_rms(pred: np.ndarray, truth: np.ndarray, bin_edges: List[float]) -> np.ndarray:
    """
    Assume pred, truth are properly scaled (e.g. linear or log)
    """
    if len(pred.shape) == 2:
        pred = pred.flatten()
        truth = truth.flatten()
    elif len(pred.shape) > 2:
        raise ValueError(f"Unsupported shape {pred.shape}")
    
    bin_indices = np.digitize(x=truth, bins=bin_edges)

    n_bins = len(bin_edges) - 1
    rms_all_bins = np.zeros(n_bins, dtype=float)
    for bin_idx in range(n_bins):
        target_locs = (bin_indices == (bin_idx + 1))
        true_bin_data = truth[target_locs]
        pred_bin_data = pred[target_locs]
        rms_bin = rms(true_bin_data, pred_bin_data)
        rms_all_bins[bin_idx] = rms_bin
    return rms_all_bins
