from __future__ import print_function, absolute_import

import numpy as np
import torch.nn.functional as F

__all__ = ['accuracy', 'cal_accuracy_each_class', 'cal_accuracy_confidence', 'cal_samples_confidence']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def precision(predict, target):
        pflat = predict.view(-1)
        tflat = target.view(-1)

        TP = (pflat * tflat).sum()

        precision = TP/pflat.sum()
        return precision

def recall(predict, target):
        pflat = predict.view(-1)
        tflat = target.view(-1)

        TP = (pflat * tflat).sum()

        recall = TP/tflat.sum()
        return recall

def JaccardIndex(predict, target):
        pflat = predict.view(-1)
        tflat = target.view(-1)

        intersection = (pflat * tflat).sum()
        union = pflat.sum() + tflat.sum()

        jaccard = intersection / (union-intersection)
        return jaccard

# calculate accuracy of each class
def cal_accuracy_each_class(output, target, topk=(1,), num_classes=100):
    """Computes the precision@k for the specified values of k"""
    temp_target = target.cpu()
    temp_output = output.cpu()
    current_target = temp_target.numpy()
    maxk = max(topk)
    batch_size = temp_target.size(0)
    _, pred = temp_output.topk(maxk, 1, True, True)
    pred = pred.t()
    res = np.zeros((num_classes, len(topk)))

    correct = pred.eq(temp_target.view(1, -1).expand_as(pred))
    # calculate accuracy of each class
    for i in range(num_classes):
        current_index = np.where( current_target == i )[0]
        if len(current_index) == 0 :
            continue
        # only calculate top1 accuracy and confident
        for k in range(len(topk)): 
            temp_correct = correct[:, current_index]
            correct_k = temp_correct[:topk[k]].view(-1).float().sum(0)
            res[i, k] = correct_k
    return res[:, 0], res[:, 1]


# calculate accuracy of each class
def cal_accuracy_confidence(output, target, num_classes=100):
    """Computes the precision@k for the specified values of k"""
    target = target.cpu().numpy()
    output = output.detach().cpu().numpy()
    confidence = np.zeros((num_classes, num_classes))
    # calculate accuracy of each class
    for i in range(num_classes):
        current_index = np.where( target == i )[0]
        if len(current_index) == 0 :
            continue
        temp = np.sum( output[current_index, :], axis=0 )
        confidence[i] = np.sum( output[current_index, :], axis=0 )
    return confidence

# calculate accuracy of each sample
def cal_samples_confidence(output):
    output = output.detach().cpu().numpy()
    return output