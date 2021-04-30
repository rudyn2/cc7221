import os
import re

import cv2
import kornia.augmentation
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import kornia.augmentation as K
import torch.nn as nn

from PIL import Image


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


class ImageDataset(Dataset):
    # MEAN = [0.485, 0.456, 0.406]
    # STD = [0.229, 0.224, 0.225]
    MEAN = [0.5457574, 0.49972787, 0.47929478]
    STD = [0.18881057, 0.18312807, 0.18848157]

    def __init__(self, path: str, width: int, height: int, process: bool = True, use_data_augmentation: bool = False):
        self.width = width
        self.height = height
        self.path = path
        self.dataset_meta = self.define_dataset_meta()
        self.image_paths = self.parse_file(path) or None
        self.image_keys = list(self.image_paths.keys())
        self.image_classes = list(self.image_paths.values())

        self.use_data_augmentation = use_data_augmentation
        self._process = process
        self.process_image_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.CenterCrop((224, 224)),
            RotationTransform(90),
            transforms.Normalize(self.MEAN, self.STD),
        ])
        self.data_aug_operations = [
            transforms.RandomRotation(degrees=[-90, 90]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),

        ]
        self.data_aug = transforms.RandomApply(self.data_aug_operations)

        self.transform = nn.Sequential(
            K.augmentation.RandomRotation(degrees=[-90, 90]),
            K.augmentation.RandomHorizontalFlip(),
            K.augmentation.RandomVerticalFlip(),
        )

        device = torch.device('cuda')


    def parse_file(self, path: str):
        if not os.path.exists(path):
            raise ValueError("Provided path doesn't exist")

        with open(os.path.join(path, self.dataset_meta), "r") as f:
            lines = f.readlines()

        paths = {}
        for line in lines:
            s = re.split(r'\t+', line.rstrip().rstrip('\t'))
            paths[s[0]] = s[1]

        return paths

    def calculate_stats(self):
        """
        Calculate the mean and standard deviation per channel over the entire dataset.
        THIS METHOD DOESN'T SAVE THE STATS
        BE CAREFULLY, THIS IS DANGEROUS CODE
        """
        r_means = []
        g_means = []
        b_means = []
        for k in tqdm(range(len(self.image_keys)), "Reading images"):
            arr = self[k][0]
            arr = arr.numpy()
            r_means.append(np.mean(arr[0, :, :]))
            g_means.append(np.mean(arr[1, :, :]))
            b_means.append(np.mean(arr[2, :, :]))

        print("Mean per channel: ", np.mean(r_means), np.mean(g_means), np.mean(b_means))
        print("Standard deviation per channel: ", np.std(r_means), np.std(g_means), np.std(b_means))

    def __len__(self):
        if self.image_paths:
            return len(self.image_paths.keys())
        raise Exception("There is no data!")

    def smart_resize(self, arr):
        ratio = self.width / min(arr.shape[:2])
        new_width = round(arr.shape[0] * ratio)
        new_height = round(arr.shape[1] * ratio)
        arr_resized = cv2.resize(arr, dsize=(int(new_height), int(new_width)))
        if arr_resized.shape[1] > arr_resized.shape[0]:
            diff = arr_resized.shape[1] - self.width
            half = int(diff / 2)
            arr_cropped = arr_resized[:, half: half + self.width]
        else:
            diff = arr_resized.shape[0] - self.width
            half = int(diff / 2)
            arr_cropped = arr_resized[half: half + self.width, :]
        return arr_cropped

    def __getitem__(self, index: int):

        assert index < len(self), f"Index must be less or equal to {len(self) - 1}"

        arr = Image.open(os.path.join(self.path, self.image_keys[index]))
        if self._process:
            arr = self.process_image_pipeline(arr)
        if self.use_data_augmentation:
            # arr2 = self.data_aug(arr)
            arr = self.transform(arr)
            arr = arr[0]

        return arr, int(self.image_classes[index])

    def define_dataset_meta(self):
        raise NotImplementedError

    def read_mapping(self):
        with open(os.path.join(self.path, "mapping.txt"), "r") as f:
            lines = f.readlines()

        mapping = {}
        for line in lines:
            s = re.split(r'\t+', line.rstrip().rstrip('\t'))
            mapping[int(s[1])] = s[0]

        return mapping


class TrainImageDataset(ImageDataset):

    def __init__(self, path: str, width: int, height: int, process: bool = True, use_data_augmentation: bool = False):
        super(TrainImageDataset, self).__init__(path, width, height, process, use_data_augmentation)

    def define_dataset_meta(self):
        return "train_sample.txt"


class TestImageDataset(ImageDataset):

    def __init__(self, path: str, width: int, height: int, process: bool = True):
        super(TestImageDataset, self).__init__(path, width, height, process, use_data_augmentation=False)

    def define_dataset_meta(self):
        return "test_sample.txt"


class ImageOfflineDataset(ImageDataset):

    def __init__(self, path: str, width: int, height: int):
        super(ImageOfflineDataset, self).__init__(path, width, height)
        self.processed_images, self.labels = self.load()

    def load(self):
        processed_images, processed_labels = [], []
        for i in tqdm(range(len(self)), "Loading images into memory "):
            arr, label = super().__getitem__(i)
            processed_images.append(arr)
            processed_labels.append(label)
        return processed_images, processed_labels

    def __getitem__(self, item: int):
        return self.processed_images[item], self.labels[item]

    def define_dataset_meta(self):
        return "train_sample.txt"


if __name__ == '__main__':

    from torch.utils.data import DataLoader

    #
    train_dataset = TrainImageDataset(r"C:\Users\aleja\Desktop\Tareas\Reconocimiento Virtual con Deep Learning\Tarea1\Imagenes\clothing-small", 224, 224)
    train_dataset.calculate_stats()
    train_dataset.read_mapping()
    # test_dataset = TestImageDataset("/home/rudy/Documents/cc7221/tarea1/data/clothing-small", 224, 224)
    print(f"Length of train dataset: {len(train_dataset)}")
    # print(f"Length of test dataset: {len(test_dataset)}")
    e = train_dataset[3]
    train_dataloader = DataLoader(train_dataset, batch_size=64, pin_memory=True, shuffle=True)
    for i in tqdm(range(len(train_dataset))):
        img, label = train_dataset[i]
        break

