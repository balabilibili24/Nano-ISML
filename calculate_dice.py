import cv2
import numpy as np
import os

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def calculate_dice():
    path = r"C:\Users\Huang Lab\Desktop\zhushi\dice"

    vessels = os.listdir(path)
    nums = len(vessels)

    means = 0.
    dices = []
    for i, vessel in enumerate(vessels):
        img1 = cv2.imread(os.path.join(path, vessel, '1', '1.1.png'), 0)    # 修改2.1
        img2 = cv2.imread(os.path.join(path, vessel, '2', 'predict_1.1.png'), 0)    # 修改2.1
        img11 = np.where(img1 > 0, 1, 0)
        img22 = np.where(img2 > 0, 1, 0)
        dice = dice_coef(img11, img22)
        means += dice
        dices.append("{} dice: {}".format(vessel, round(dice, 4)))

    print("means dices: ", means / nums)
    print(dices)


# note: 如果测试预测的mask和gt(金标准)， img1: pre_mask    img2: gt
def calculate_F1_score():
    path = r"C:\Users\Huang Lab\Desktop\zhushi\dice"

    vessels = os.listdir(path)
    nums = len(vessels)

    means = 0.
    F1_scores = []
    for i, vessel in enumerate(vessels):
        img1 = cv2.imread(os.path.join(path, vessel, '1', '1.1.png'), 0)  # 修改2.1
        img2 = cv2.imread(os.path.join(path, vessel, '2', 'predict_1.1.png'), 0)  # 修改2.1
        img11 = np.where(img1 > 0, 1, 0)
        img22 = np.where(img2 > 0, 1, 0)

        TP = img11 * img22
        FP = img11 - TP
        FN = img22 - TP
        precision = TP.sum() / (TP.sum() + FP.sum())
        recall = TP.sum() / (TP.sum() + FN.sum())
        F1 = (2 * precision * recall) / (precision + recall)
        means += F1
        F1_scores.append("{} F1_score: {}".format(vessel, round(F1, 4)))

    print("means F1_score: ", means / nums)
    print(F1_scores)


def calculate_jaccard():
    path = r"C:\Users\Huang Lab\Desktop\zhushi\dice"

    vessels = os.listdir(path)
    nums = len(vessels)

    means = 0.
    jaccard_scores = []
    for i, vessel in enumerate(vessels):
        img1 = cv2.imread(os.path.join(path, vessel, '1', '1.1.png'), 0)  # 修改2.1
        img2 = cv2.imread(os.path.join(path, vessel, '2', 'predict_1.1.png'), 0)  # 修改2.1
        img11 = np.where(img1 > 0, 1, 0)
        img22 = np.where(img2 > 0, 1, 0)

        y_true_f = img11.flatten()
        y_pred_f = img22.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        jaccard_score = (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)

        means += jaccard_score
        jaccard_scores.append("{} jaccard_score: {}".format(vessel, round(jaccard_score, 4)))

    print("means jaccard_score: ", means / nums)
    print(jaccard_scores)


if __name__ == '__main__':
    calculate_dice()
    calculate_jaccard()
