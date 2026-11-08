import numpy as np

def voxelwise_pearsonr(pred, gt, eps=1e-12):
    """
    Compute Pearson correlation across images for each voxel.

    Parameters
    ----------
    pred : np.ndarray, shape (N, D)
        Predicted fMRI responses.
    gt : np.ndarray, shape (N, D)
        Ground-truth fMRI responses.
    eps : float
        Minimum valid denominator.

    Returns
    -------
    mean_corr : float
        Mean correlation across valid voxels.
    voxel_corr : np.ndarray, shape (D,)
        Pearson correlation for each voxel.
        Zero-variance voxels are assigned NaN.
    """
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: pred has shape {pred.shape}, "
            f"but gt has shape {gt.shape}."
        )

    pred = pred - pred.mean(axis=0, keepdims=True)
    gt = gt - gt.mean(axis=0, keepdims=True)

    numerator = np.sum(pred * gt, axis=0)
    denominator = np.sqrt(
        np.sum(pred ** 2, axis=0) *
        np.sum(gt ** 2, axis=0)
    )

    voxel_corr = np.divide(
        numerator,
        denominator,
        out=np.full(pred.shape[1], np.nan),
        where=denominator > eps
    )

    mean_corr = np.nanmean(voxel_corr)

    return mean_corr, voxel_corr
