""" This script evaluates the models using IoU and Fwbeta measures between the predictions and the annotations. """
import argparse
import os
import cv2
import numpy as np

from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from src.utils.utils import get_data_frame, weighted_f_beta_score


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str,
                        default="...",
                        )
    parser.add_argument('--ann_dir', type=str,
                        default="...",
                        )
    parser.add_argument('--task', type=int, default=2)  # 1: Fwbeta score, 2: Masked IoU
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--visualise', type=bool, default=False)
    parser.add_argument('--save_res', type=bool, default=False)
    parser.add_argument('--dest_path', type=str,
                        default="..."
                        )
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    pred_dir = args.pred_dir
    ann_dir = args.ann_dir
    task = args.task
    assert task in [1, 2], "Task number not valid"
    save_res = args.save_res
    dest_path = args.dest_path
    num_classes = args.num_classes

    print("==============")
    print("pred_dir: ", pred_dir)
    print("ann_dir: ", ann_dir)
    print("task: ", task)
    print("num_classes: ", num_classes)
    print("save_res: ", save_res)
    print("dest_path: ", dest_path)
    print("==============")

    # Retrieve predictions and annotations path
    preds_name = os.listdir(pred_dir)
    anns_name = os.listdir(ann_dir)
    preds_name.sort()
    anns_name.sort()

    assert len(preds_name) == len(anns_name), "Predictions and annotations have different number of elements"

    # Initialise variables
    res = get_data_frame(len(preds_name), num_classes)
    num_info = 9

    for i in tqdm(range(len(preds_name))):
        filename = os.path.basename(preds_name[i])
        pred = cv2.imread(os.path.join(pred_dir, filename), cv2.IMREAD_GRAYSCALE)
        ann = cv2.imread(os.path.join(ann_dir, filename), cv2.IMREAD_GRAYSCALE)

        # Save per class result
        for c in range(0, num_classes):
            # Compute measures
            if task == 1:
                TPw, FPw, FNw, Q = [0, 0, 0, 0]

                pred_tmp = pred.copy().astype(float)
                ann_tmp = ann.copy().astype(float)

                # Consider '1' entries equal to the class
                pred_tmp[pred == c] = 1
                ann_tmp[ann == c] = 1

                # Consider '0' other entries
                pred_tmp[pred != c] = 0
                ann_tmp[ann != c] = 0

                if np.count_nonzero(ann_tmp) >= 1:
                    # Compute Fwbeta
                    B, Et, EA, R, P, TPw, FPw, FNw, Q = weighted_f_beta_score(ann_tmp, pred_tmp)
                else:
                    TPw, FPw, FNw, Q = [-1, -1, -1, -1]

                res.iloc[i, c * num_info + 6] = TPw
                res.iloc[i, c * num_info + 7] = FPw
                res.iloc[i, c * num_info + 8] = FNw
                res.iloc[i, c * num_info + 9] = Q

            if task == 2:
                tp, fp, fn, tn, iou = [0, 0, 0, 0, 0]

                pred_tmp = pred.flatten()
                ann_tmp = ann.flatten()

                # Consider '1' entries equal to the class
                pred_tmp[pred.flatten() == c] = 1
                ann_tmp[ann.flatten() == c] = 1

                # Consider '0' other entries
                pred_tmp[pred.flatten() != c] = 0
                ann_tmp[ann.flatten() != c] = 0
                if np.count_nonzero(pred_tmp) == 0 and np.count_nonzero(ann_tmp) == 0:
                    tn = confusion_matrix(ann_tmp, pred_tmp).ravel()[0]
                elif np.count_nonzero(pred_tmp) == len(pred_tmp) and np.count_nonzero(ann_tmp) == len(ann_tmp):
                    tp = confusion_matrix(ann_tmp, pred_tmp).ravel()[0]
                    iou = tp / (tp + fp + fn)
                else:
                    tn, fp, fn, tp = confusion_matrix(ann_tmp, pred_tmp).ravel()
                    iou = tp / (tp + fp + fn)

                res.iloc[i, c * num_info + 1] = tp
                res.iloc[i, c * num_info + 2] = fp
                res.iloc[i, c * num_info + 3] = fn
                res.iloc[i, c * num_info + 4] = tn
                res.iloc[i, c * num_info + 5] = iou

            # Fill table
            res.iloc[i, 0] = filename

    if task == 1:
        fwbeta = np.zeros(num_classes - 1)
        eps = np.spacing(1)
        # No background
        for c in range(1, num_classes):
            no_minusone = res.iloc[:, c * num_info + 9] != -1
            num_el = len(res.iloc[:, c * num_info + 9][no_minusone])
            if num_el == 0:
                fwbeta[c - 1] = 0
            else:
                fwbeta[c - 1] = res.iloc[:, c * num_info + 9][no_minusone].sum() / num_el
        print('Class Fwbeta total:', ' '.join(f'{x * 100:.2f}' for x in fwbeta),
              f'  |  Mean Fwbeta: {fwbeta.mean() * 100:.2f}')

    elif task == 2:
        iou = np.zeros(num_classes - 1)
        pr = np.zeros(num_classes - 1)
        rc = np.zeros(num_classes - 1)
        tp = np.zeros(num_classes - 1)
        fp = np.zeros(num_classes - 1)
        fn = np.zeros(num_classes - 1)
        # No background
        for c in range(1, num_classes):  # no background
            iou[c - 1] = res.iloc[:, c * num_info + 1].sum() / (
                    res.iloc[:, c * num_info + 1].sum() + res.iloc[:, c * num_info + 2].sum() + res.iloc[:,
                                                                                                c * num_info + 3].sum())
            if res.iloc[:, c * num_info + 2].sum() == 0 and res.iloc[:, c * num_info + 1].sum() == 0:
                pr[c - 1] = -1
            else:
                pr[c - 1] = res.iloc[:, c * num_info + 1].sum() / (
                        res.iloc[:, c * num_info + 1].sum() + res.iloc[:, c * num_info + 2].sum())
            rc[c - 1] = res.iloc[:, c * num_info + 1].sum() / (
                    res.iloc[:, c * num_info + 1].sum() + res.iloc[:, c * num_info + 3].sum())
        print('Class P total:', ' '.join(f'{x * 100:.2f}' for x in pr), f'  |  Mean P: {pr.mean() * 100:.2f}')
        print('Class R total:', ' '.join(f'{x * 100:.2f}' for x in rc), f'  |  Mean R: {rc.mean() * 100:.2f}')
        print('Class IoU total:', ' '.join(f'{x * 100:.2f}' for x in iou), f'  |  Mean IoU: {iou.mean() * 100:.2f}')

    if save_res:
        res.to_csv(dest_path, index=False)
