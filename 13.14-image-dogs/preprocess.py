import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils.data import Subset

data_dir = './dataset/imagenet-dog'
valid_ratio = 0.1

def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

if __name__ == '__main__':
    reorg_dog_data(data_dir, valid_ratio)

def load_argumentation(data_dir, train_length, batch_size = 256, num_workers = 4):
    transform_train = torchvision.transforms.Compose([
        # 随机裁剪图像，所得图像为原始面积的0.08～1之间，高宽比在3/4和4/3之间。
        # 然后，缩放图像以创建224x224的新图像
        torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                ratio=(3.0/4.0, 4.0/3.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        # 随机更改亮度，对比度和饱和度
        torchvision.transforms.ColorJitter(brightness=0.4,
                                        contrast=0.4,
                                        saturation=0.4),
        # 添加随机噪声
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # 从图像中心裁切224x224大小的图片
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]

    valid_ds, test_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]

    train_length = len(train_ds)
    print(f'训练样本为前{train_length}个')

    train_iter, train_valid_iter = [torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True, num_workers = num_workers)
        for dataset in (Subset(train_ds, range(train_length)), train_valid_ds)]

    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                            drop_last=True, num_workers = num_workers)

    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                            drop_last=False, num_workers = num_workers)
    return train_iter, train_valid_iter, valid_iter, test_iter, train_valid_ds, test_ds