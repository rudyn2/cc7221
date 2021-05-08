import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from os import listdir
from os.path import join, isdir
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import random


class SketchDataset(ABC, Dataset):
    # MEAN = [0.9818744]
    # STD = [0.0077032577]

    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]

    def __init__(self, path: str):
        self._path = path
        self._images = self._read(os.path.join(self._path, self.get_txt_name()))  # image_path -> class
        self._images_paths = list(self._images.keys())  # image_path
        self._class_mapping_inverted = self._read(join(self._path, "mapping.txt"))  # class label -> class number
        self._class_mapping = {v: k for k, v in self._class_mapping_inverted.items()}   # class number -> class label
        self._class_groups = self._build_groups()  # class label -> list with group of image paths

        self.process_image_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            RotationTransform(90),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            # transforms.Normalize(self.mean, self.std),
        ])

    def _build_groups(self):
        groups = defaultdict(list)
        for img_path, class_number in self._images.items():
            groups[self._class_mapping[class_number]].append(join(self._path, img_path))
        return groups

    @abstractmethod
    def get_txt_name(self):
        raise NotImplementedError

    @staticmethod
    def _read(file_path: str) -> dict:
        with open(file_path, "r") as f:
            lines = f.readlines()

        paths = {}
        for line in lines:
            s = re.split(r'\t+', line.rstrip().rstrip('\t'))
            paths[s[0]] = int(s[1])
        return paths

    def __getitem__(self, item):
        img_path = self._images_paths[item]
        img_class = self._images[img_path]
        img_path = os.path.join(self._path, img_path)
        img = Image.open(img_path)
        img = self.process_image_pipeline(img)
        return img, img_class

    def __len__(self):
        return len(self._images_paths)

    def calculate_stats(self):
        means = []
        for k in tqdm(range(len(self._images_paths)), "Reading images"):
            arr = self[k][0]
            arr = arr.numpy()
            means.append(np.mean(arr[0, :, :]))

        print("Mean: ", np.mean(means))
        print("Standard deviation: ", np.std(means))

    @property
    def class_groups(self):
        return self._class_groups

    @property
    def images(self):
        return self._images

    @property
    def class_mapping(self):
        return self._class_mapping

    @property
    def path(self):
        return self._path


class SketchTrainDataset(SketchDataset):

    def __init__(self, path: str):
        super(SketchTrainDataset, self).__init__(path)

    def get_txt_name(self):
        return "train.txt"


class SketchTestDataset(SketchDataset):

    def __init__(self, path: str):
        super(SketchTestDataset, self).__init__(path)

    def get_txt_name(self):
        return "test.txt"


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


class FlickrDataset(Dataset):
    def __init__(self, path: str):
        super(FlickrDataset, self).__init__()
        self._path = path
        self._class_mapping = {}            # class number -> class label

        self._images = {}
        self.read()

        self._class_groups = self._build_groups()
        self._images_path = list(self._images.keys())
        self.process_image_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            RotationTransform(90),
            # transforms.Normalize(self.mean, self.std),
        ])

    def _build_groups(self):
        groups = defaultdict(list)
        for img_path, class_number in self._images.items():
            groups[self._class_mapping[class_number]].append(join(self._path, img_path))
        print(f"{len(groups)} classes with a total of {sum([len(g) for g in groups.values()])} samples")
        return groups

    def read(self):
        folders = [f for f in listdir(self._path) if isdir(join(self._path, f))]
        for i, folder in enumerate(folders):
            self._class_mapping[i] = folder
            for image in listdir(join(self._path, folder)):
                full_path = join(self._path, folder, image)
                self._images[full_path] = i

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        img_path = self._images_path[item]
        img_class = self._images[img_path]
        img = Image.open(img_path)
        img = self.process_image_pipeline(img)
        return img, img_class

    def __len__(self) -> int:
        return len(self._images_path)

    def get_random_from_class(self, class_label: str) -> str:
        print(len(self._class_groups))
        class_group = self._class_groups[class_label]
        sample = random.sample(class_group, 1)[0]
        return sample

    def get_random_from_non_class(self, class_label: str) -> str:
        negative_groups = [gk for gk in self._class_groups.keys() if gk != class_label]
        negative_class = random.sample(negative_groups, 1)[0]
        return self.get_random_from_class(negative_class)

    @property
    def class_mapping(self):
        return self._class_mapping

    @property
    def images(self):
        return self._images


class ContrastiveDataset(Dataset):
    def __init__(self, flickr_dataset: FlickrDataset, sketches_dataset: SketchDataset):
        super(ContrastiveDataset, self).__init__()
        self._flickr = flickr_dataset
        self._sketches = sketches_dataset
        self._pairs = self._create_pairs()

    def _create_pairs(self):
        pairs = []
        for img_path, class_number in self._flickr.images.items():
            flickr_class_label = self._flickr.class_mapping[class_number]
            for second_img_path in self._sketches.class_groups[flickr_class_label]:
                pairs.append((img_path, second_img_path))
        return pairs

    def __getitem__(self, item):
        # we took an image from flickr dataset and then we uniformly sample an sketch from sketches dataset
        img_flickr_path, img_sketches_path = self._pairs[item]    # (flickr, sketch)
        img_flickr, img_sketches = Image.open(img_flickr_path), Image.open(img_sketches_path)
        img_flickr = self._flickr.process_image_pipeline(img_flickr)
        img_sketches = self._sketches.process_image_pipeline(img_sketches)
        return img_flickr, img_sketches

    def __len__(self):
        return len(self._pairs)


class TripletDataset(Dataset):
    def __init__(self, flickr_dataset: FlickrDataset, sketches_dataset: SketchDataset):
        super(TripletDataset, self).__init__()
        self._flickr = flickr_dataset
        self._sketches = sketches_dataset
        self._pairs = self._create_triplets()

    def _create_triplets(self):
        triplets = []
        for anchor_path, class_number in tqdm(self._sketches.images.items(), "Creating triplet dataset"):
            # get an anchor
            anchor_path = join(self._sketches.path, anchor_path)
            anchor_class_label = self._sketches.class_mapping[class_number]
            positive_path = self._flickr.get_random_from_class(anchor_class_label)
            negative_path = self._flickr.get_random_from_non_class(anchor_class_label)
            triplets.append((anchor_path, positive_path, negative_path))

            # # for each positive sample
            # for positive_path in self._flickr._class_groups[anchor_class_label]:
            #     # this is an expensive iteration
            #     negative_groups = [gk for gk in self._flickr._class_groups.keys() if gk != anchor_class_label]
            #     negative_groups = random.sample(negative_groups, 1)
            #     for negative_group in negative_groups:
            #         negative_samples = random.sample(self._flickr._class_groups[negative_group], 3)
            #         for negative_path in negative_samples:
            #             triplets.append((anchor_path, positive_path, negative_path))

        return triplets

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_flickr_dataset = FlickrDataset('/home/rudy/Documents/cc7221/tarea2/data/Flickr25K')
    train_flickr_loader = DataLoader(train_flickr_dataset, batch_size=8)
    for images, labels in train_flickr_loader:
        print(images.shape)
        break

    train_sketches_dataset = SketchTrainDataset('/home/rudy/Documents/cc7221/tarea2/data/Sketch_EITZ')
    train_loader = DataLoader(train_sketches_dataset, batch_size=8)
    for images, labels in train_loader:
        print(images.shape)
        break

    # train_contrastive = ContrastiveDataset(train_flickr_dataset, train_sketches_dataset)
    # d = train_contrastive[0]
    # train_contrastive_loader = DataLoader(train_contrastive, batch_size=8)
    # for images, labels in train_contrastive_loader:
    #     print(images.shape)
    #     break

    print("Creating triplet dataset ...")
    train_triplet = TripletDataset(train_flickr_dataset, train_sketches_dataset)
    d = train_triplet[0]
