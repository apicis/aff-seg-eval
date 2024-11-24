import numpy as np
import scipy
import cv2
import pandas as pd


def compute_IoU(cm):
    """
    Adapted from:
        https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py
        https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/metrics.py#L2716-L2844
    """

    sum_over_row = cm.sum(axis=0)
    sum_over_col = cm.sum(axis=1)
    true_positives = np.diag(cm)

    # sum_over_row + sum_over_col = 2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    iou = true_positives / denominator

    return iou, np.nanmean(iou)


def gaussian_filter(kernel_size, sigma=5, muu=0):
    """ Creates a gaussian filter with specified kernel size and standard deviation.

    :param kernel_size: size of the kernel of the gaussian filter
    :param sigma: standard deviation
    :param muu: mean
    :return: gaussian filter
    """
    # Initializing value of x,y as grid of kernel size in the range of kernel size
    start = end = (kernel_size - 1) // 2
    x, y = np.meshgrid(np.linspace(-start, end, kernel_size), np.linspace(-start, end, kernel_size))
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def weighted_f_beta_score(gt, pred, beta=1.0, visualise=False):
    """ Compute the Weighted F-beta measure (as proposed in "How to Evaluate Foreground Maps?" [Margolin et al. - CVPR'14])
    Original MATLAB source code from: [https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/FGEval/resources/WFb.m]
    :param gt: Binary ground truth map
    :param pred: FG - Binary/Non binary candidate map with values in the range [0-1]
    :param beta: attach 'beta' times as much importance to Recall as to Precision (default=1)
    :result: the Weighted F-beta score
    """
    if visualise:
        cv2.imshow("GT", (gt * 255).astype(np.uint8))
        cv2.imshow("PRED", (pred * 255).astype(np.uint8))
        cv2.waitKey(0)

    if np.min(pred) < 0.0 or np.max(pred) > 1.0:
        raise ValueError("'candidate' values must be inside range [0 - 1]")

    if gt.dtype in [bool]:
        gt_mask = gt
        not_gt_mask = np.logical_not(gt_mask)
        gt = np.array(gt, dtype=float)
    else:
        gt_mask = gt == 1
        not_gt_mask = np.logical_not(gt)

    # E is the absolute difference between the prediction and the ground truth
    E = np.abs(pred - gt)

    if visualise:
        cv2.imshow("E", (E * 255).astype(np.uint8))
        temp = E * gt
        cv2.imshow("False negative pixels", (temp * 255).astype(np.uint8))
        cv2.waitKey(0)

    # Compute the euclidean distance transform of the binary ground truth (gt)
    # For each pixel in gt, the distance transform assigns a number that is the distance between that pixel
    # and the nearest nonzero pixel of gt.
    # distance_transform_edt finds the distance to the nearest zero pixel, that is why we pass not_gt_mask
    dist, idx = scipy.ndimage.distance_transform_edt(not_gt_mask, return_indices=True)

    if visualise:
        dist_temp = dist / np.amax(dist)
        cv2.imshow("Dist", cv2.applyColorMap((dist_temp * 255).astype(np.uint8), cv2.COLORMAP_JET))
        cv2.waitKey(0)

    # Pixel dependency
    size = 7
    sigma = 5
    K = gaussian_filter(size, sigma)
    Et = np.array(E)
    Et[not_gt_mask] = E[
        idx[0, not_gt_mask], idx[1, not_gt_mask]]  # To deal correctly with the edges of the foreground region

    if visualise:
        Et_temp = Et / np.amax(Et)
        Et_temp_vis = cv2.applyColorMap((Et_temp * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow("Et", Et_temp_vis)
        # temp = np.zeros_like(Et_temp_vis)
        # temp[gt == 1] = np.array([255, 255, 255]).astype((np.uint8))
        # cv2.imshow("Et with W", cv2.addWeighted(Et_temp_vis, 1.0, temp, 0.6, 0.0))
        cv2.waitKey(0)

    EA = scipy.signal.convolve2d(Et, K, mode='same')
    if visualise:
        EA_temp = EA / np.amax(EA)
        EA_vis = cv2.applyColorMap((EA_temp * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow("EA", EA_vis)
        # EA_vis[E == 1] = np.array([255, 255, 255]).astype((np.uint8))
        # cv2.imshow("EA with E", EA_vis)
        cv2.waitKey(0)

    min_E_EA = E.copy().astype(float)
    min_E_EA[np.logical_and(gt == 1, EA < E)] = EA[np.logical_and(gt == 1, EA < E)]

    if visualise:
        min_E_EA_temp = min_E_EA / np.amax(min_E_EA)
        min_E_EA_vis = cv2.applyColorMap((min_E_EA_temp * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow("min_E_EA", min_E_EA_vis)
        cv2.waitKey(0)

    # Pixel importance
    B = np.ones(gt.shape)
    B[not_gt_mask] = 2 - np.exp(np.log(1 - 0.5) / 5 * dist[not_gt_mask])

    if visualise:
        B_temp = B / np.amax(B)
        B_temp = cv2.applyColorMap((B_temp * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow("B", (B_temp).astype(np.uint8))
        cv2.waitKey(0)

    Ew = min_E_EA * B
    if visualise:
        Ew_temp = Ew / np.amax(Ew)
        cv2.imshow("Ew", cv2.applyColorMap((Ew_temp * 255).astype(np.uint8), cv2.COLORMAP_JET))
        cv2.waitKey(0)

    # Final metric computation
    eps = np.spacing(1)
    TPw = np.sum(gt) - np.sum(Ew[gt_mask])
    FPw = np.sum(Ew[not_gt_mask])
    FNw = np.sum(Ew[gt_mask])
    R = 1 - np.mean(Ew[gt_mask])  # Weighed Recall
    P = TPw / (eps + TPw + FPw)  # Weighted Precision

    # Q = 2 * (R * P) / (eps + R + P)  # Beta=1
    Q = (1 + beta ** 2) * (R * P) / (eps + R + (beta * P))

    return B, Et, EA, R, P, TPw, FPw, FNw, Q


def get_data_frame(num_samples, num_classes):
    """ Creates a dataframe of specified number of samples and number of affordance classes.

    :param num_samples: how many rows in the dataframe
    :param num_classes: how many affordance classes
    :return: dataframe
    """
    cols_headers = ['Image']

    for c in range(0, num_classes):
        cols_headers = cols_headers + ['TP{}'.format(c), 'FP{}'.format(c), 'FN{}'.format(c), 'TN{}'.format(c),
                                       'IOU{}'.format(c), 'TPw{}'.format(c), 'FPw{}'.format(c), 'FNw{}'.format(c),
                                       'FWB{}'.format(c)]
    # Initialise
    dummy_mat = -np.ones((num_samples, len(cols_headers)))

    # Build data frame
    df = pd.DataFrame(data=dummy_mat, columns=cols_headers)

    return df
