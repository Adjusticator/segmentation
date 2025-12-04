import numpy as np
import torch
#计算每个类别的IOU并求平均
def iou(pred, target, n_classes = 21):
    pred_mask = torch.argmax(pred, 1)
    score = 0.0
    num = 0
    for i in range(n_classes):
      tp_mask = torch.logical_and(pred_mask == target, pred_mask == i)
      tp = torch.where(tp_mask, 1, 0).sum(dtype=torch.float)
      fp = torch.where(pred_mask == i, 1, 0).sum(dtype=torch.float)
      fn = torch.where(target == i, 1, 0).sum(dtype=torch.float)
      if fp+fn-tp != 0:
        score += (tp/(fp+fn-tp))
        num += 1
    return (score/num)

#计算整体的像素的正确率
def pixel_acc(pred, target):
    pred_mask = torch.argmax(pred, 1)
    bin_mask = torch.where(pred_mask == target, 1, 0).mean(dtype=torch.float)
    return bin_mask