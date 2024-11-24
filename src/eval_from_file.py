""" This script computes performance measures from cvs results file. """
import argparse
import numpy as np
import pandas as pd


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_file', type=str,
                        default="...",
                        )
    parser.add_argument('--task', type=int, default=2)  # 1: Fwbeta score, 2: Masked IoU
    parser.add_argument('--num_classes', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    res_file = args.res_file
    task = args.task
    assert task in [1, 2], "Task number not valid"
    num_classes = args.num_classes

    # Initialise variables
    res = pd.read_csv(res_file)
    num_info = 9

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
            num_el = len(res.iloc[:, c * num_info + 9])
            # print(num_el)
            iou[c - 1] = res.iloc[:, c * num_info + 1].sum() / (
                    res.iloc[:, c * num_info + 1].sum() + res.iloc[:, c * num_info + 2].sum() + res.iloc[:,
                                                                                                c * num_info + 3].sum())
            pr[c - 1] = res.iloc[:, c * num_info + 1].sum() / (
                    res.iloc[:, c * num_info + 1].sum() + res.iloc[:, c * num_info + 2].sum())
            rc[c - 1] = res.iloc[:, c * num_info + 1].sum() / (
                    res.iloc[:, c * num_info + 1].sum() + res.iloc[:, c * num_info + 3].sum())
        print('Class P total:', ' '.join(f'{x * 100:.2f}' for x in pr), f'  |  Mean P: {pr.mean() * 100:.2f}')
        print('Class R total:', ' '.join(f'{x * 100:.2f}' for x in rc), f'  |  Mean R: {rc.mean() * 100:.2f}')
        print('Class IoU total:', ' '.join(f'{x * 100:.2f}' for x in iou), f'  |  Mean IoU: {iou.mean() * 100:.2f}')
