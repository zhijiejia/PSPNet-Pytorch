import os
import os.path
import cv2
import numpy as np
from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    '''
        :param data_root:
        :param data_list: the txt file path
        :return:
    '''
    assert split in ['train', 'val', 'test']
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        if split == 'test':
            image_name = '/root/PycharmProjects/pythonProject/VOCdevkit/VOC2012/JPEGImages/' + line + '.jpg'
            label_name = '/root/PycharmProjects/pythonProject/VOCdevkit/VOC2012/SegmentationClass/' + line + '.png'
        else:
            image_name = '/root/PycharmProjects/pythonProject/VOCdevkit/VOC2012/JPEGImages/' + line + '.jpg'
            label_name = '/root/PycharmProjects/pythonProject/VOCdevkit/VOC2012/SegmentationClass/' + line + '.png'
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemDataset(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        image, label = self.transform(image, label)
        return image, label


if __name__ == '__main__':
    #SemDataset(split='train', data_root='/root/PycharmProjects/pythonProject/VOCdevkit/VOC2012/ImageSets', data_list='/root/PycharmProjects/pythonProject/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval_aug.txt')
    SemDataset(split='test', data_root='/root/PycharmProjects/pythonProject/VOCdevkit/VOC2012/', data_list='/root/PycharmProjects/pythonProject/VOCdevkit/VOC2012/ImageSets/Segmentation/test_aug.txt')