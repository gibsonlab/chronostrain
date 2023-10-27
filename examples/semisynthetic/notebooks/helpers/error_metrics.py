import numpy as np
import scipy.integrate
import scipy.stats


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
    if len(pred.shape) == 2:
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
            res = scipy.stats.spearmanr(pred_t, truth_t)
            timeseries_corrs.append(res.correlation)
        elif len(pred_t.shape) == 2:
            stats = np.array([
                scipy.stats.spearmanr(pred_t_sample, truth_t).correlation
                for pred_t_sample in pred_t
            ])
            timeseries_corrs.append(stats)
        else:
            raise ValueError("Unrecognized pred shape {}".format(pred.shape))
    timeseries_corrs = np.stack(timeseries_corrs, axis=0)  # shape is (T) or (T,N)
    return np.mean(timeseries_corrs, axis=0)
