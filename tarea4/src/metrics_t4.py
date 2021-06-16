import numpy as np


def dice(pred, true, k=1):
    dice0 = []
    dice1 = []
    dice2 = []
    pred = np.int32(np.round(pred))
    true = np.int32(np.round(true))

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):

            intersection = np.sum((pred[i, j, :, :] & true[i, j, :, :])) * 2.0
            dice = intersection / (np.sum(pred[i, j, :, :]) + np.sum(true[i, j, :, :]))

            if j == 0:
                dice0.append(dice)
            if j == 1:
                dice1.append(dice)
            if j == 2:
                dice2.append(dice)
    dice_f = np.mean([dice0, dice1, dice2])

    return dice_f, np.mean(dice0), np.mean(dice1), np.mean(dice2)


def get_IoU_f(pred, true):
    c = 1e-6
    iou0 = []
    iou1 = []
    iou2 = []
    pred = np.int32(np.round(pred))
    true = np.int32(np.round(true))

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):

            intersection = (pred[i, j, :, :] & true[i, j, :, :]).sum(1)
            union = (pred[i, j, :, :] | true[i, j, :, :]).sum(1)

            iou = (intersection + c) / (union + c)  # sumamos c para no 0/0
            if j == 0:
                iou0.append(iou.mean())
            if j == 1:
                iou1.append(iou.mean())
            if j == 2:
                iou2.append(iou.mean())
    iou_f = np.mean([iou0, iou1, iou2])

    return iou_f, np.mean(iou0), np.mean(iou1), np.mean(iou2)
