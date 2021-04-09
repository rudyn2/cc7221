import os
import re

import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class ImageDataset(Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, path: str, width: int, height: int):
        super(Dataset, self).__init__()

        self.path = path
        self.image_paths = self.parse_file(path) or None
        self.image_keys = list(self.image_paths.keys())
        self.image_classes = list(self.image_paths.values())
        self.width = width
        self.height = height

    def parse_file(self, path: str):
        if not os.path.exists(path):
            raise ValueError("Provided path doesn't exist")

        with open(os.path.join(path, "train_sample.txt"), "r") as f:
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

        train_stacked = []
        for k in tqdm(range(len(self.image_keys)), "Reading images"):
            arr = self[k]
            train_stacked.append(arr)

        train_stacked = np.stack(train_stacked)
        means = [np.mean(train_stacked[:, :, :, i]) for i in range(2)]
        stds = [np.std(train_stacked[:, :, :, i]) for i in range(2)]

        print("Mean per channel: ", means)
        print("Standard deviation per channel: ", stds)

    def __len__(self):
        if self.image_paths:
            return len(self.image_paths.keys())
        raise Exception("There is no data!")

    def __getitem__(self, index: int):

        assert index < len(self), f"Index must be less or equal to {len(self) - 1}"

        arr = cv2.imread(os.path.join(self.path, self.image_keys[index]), cv2.IMREAD_COLOR)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) / 255.0
        for c in range(2):
            arr[:, :, c] = (arr[:, :, c] - self.MEAN[c]) / self.STD[c]

        arr = cv2.resize(arr, dsize=(self.width, self.height))
        arr = np.swapaxes(arr, 0, -1)
        # arr = self.transform(arr)
        return arr, float(self.image_classes[index])


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


if __name__ == '__main__':
    ImageOfflineDataset(r"C:\Users\C0101\PycharmProjects\cc7221\data\clothing-small", 224, 224)