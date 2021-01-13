import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            # 设置所有CPU的随机种子
    torch.cuda.manual_seed(seed)       # 设置所有GPU的随机种子
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

'''
    1. This file provide the paint function for the output of NetWork
    2. NetWork can predict 32 classes, therefore the length of colormap is 32
'''

colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

def decode_segmap(label_mask, classes=0):
    if classes == 0:
        raise Exception("The classes are illegal!")
    img_height = label_mask.shape[0]
    img_width = label_mask.shape[1]
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, classes):
        r[label_mask == ll] = colormap[ll][0]
        g[label_mask == ll] = colormap[ll][1]
        b[label_mask == ll] = colormap[ll][2]
    rgb = np.zeros((img_height, img_width, 3))
    # print(r.shape)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb.astype(np.uint8)


def show_pred(output, label, writer, cnt):
    output = output.cpu().data.numpy()
    for j in range(6):
        img = decode_segmap(output[j, ...], 21)  # 473 * 473 * 3
        img = torch.tensor(img.transpose((2, 0, 1)))  # 3 * 473 * 473
        writer.add_image(tag=f'epoch-pred-{cnt}-{j}', img_tensor=img)
        writer.add_image(tag=f'epoch-pred-{cnt}-{j}-label', img_tensor=torch.unsqueeze(label[j, ...] * 4, dim=0))
