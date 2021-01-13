import torch
import numpy as np

def batch_pix_accuracy(predict, target):

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_intersection_union(predict, target, num_class):
    '''
    if the label shape == (N, C, H, W), do squeeze for label, make it's shape to (N, H, W), which shape equal to output's shape size
    :param output:  N * H * W
    :param target:  N * H * W
    :param num_class: 21
    :return:
    '''

    if len(predict.shape) == 4:
        predict = torch.argmax(predict, dim=1)

    if len(target.shape) == 4:
        target = torch.squeeze(target, dim=1)

    predict = predict + 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def eval_metrics(output, target, num_classes):
    correct, labeled = batch_pix_accuracy(output.data, target)
    inter, union = batch_intersection_union(output.data, target, num_classes)
    return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5)]  # round(list, 5) mean keep 5 digits after point for everyone in list

# 设标签宽W，长H
def fast_hist(a, b, n):
    # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)
    k = (a >= 0) & (a < n)
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    # 返回中，写对角线上的为分类正确的像素点
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def per_class_PA(hist):
    # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
    return np.diag(hist) / hist.sum(1)


def compute_mIoU(labels, preds, num_classes=21):
    # 计算mIoU的函数

    if len(preds.shape) == 4:
        preds = torch.argmax(preds, dim=1)

    if len(labels.shape) == 4:
        labels = torch.squeeze(labels, dim=1)

    hist = np.zeros((num_classes, num_classes))

    n, h, w = labels.shape

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    for ind in range(n):
        # 读取一张图像分割结果
        pred = preds[ind]
        # 读取一张对应的标签
        label = labels[ind]

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):
            print('label and pred shape not equal!')
            continue

        # 对一张图片计算21×21的hist矩阵，并累加
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

    return hist


def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice系数
    # --------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score
