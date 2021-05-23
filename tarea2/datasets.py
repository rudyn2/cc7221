import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from os import listdir
from os.path import join, isdir
from typing import Tuple

import numpy as np
import torch
from utils import ToRGB, RotationTransform
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import random
import glob


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
        self._class_mapping_inverted = self._patch(self._class_mapping_inverted)
        self._class_mapping = {v: k for k, v in self._class_mapping_inverted.items()}   # class number -> class label
        self._class_groups = self._build_groups()  # class label -> list with group of image paths

        self.process_image_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            RotationTransform(90),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            # transforms.Normalize(self.mean, self.std),
        ])

    @staticmethod
    def _patch(input_dict: dict) -> dict:
        old, new = input_dict, {}
        for k, v in old.items():
            new[str(k).replace('-', '_')] = v
        return new

    def _build_groups(self):
        groups = defaultdict(list)
        for img_path, class_number in self._images.items():
            group_label = self._class_mapping[class_number]
            groups[group_label].append(join(self._path, img_path))
        print(f"[SKETCHES] {len(groups)} classes with a total of {sum([len(g) for g in groups.values()])} samples")
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

    @property
    def class_mapping_inverted(self):
        return self._class_mapping_inverted


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


class FlickrDataset(Dataset):
    def __init__(self, path: str, class_mapping: dict):
        super(FlickrDataset, self).__init__()
        self._path = path

        self._images = {}
        self.read()
        self._class_mapping = class_mapping              # class number -> class label
        self._class_mapping_inverted = {v: k for k, v in self._class_mapping.items()}

        self._class_groups = self._build_groups()
        self._images_path = list(self._images.keys())
        self.process_image_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            RotationTransform(90),
            ToRGB(),
            # transforms.Normalize(self.mean, self.std),
        ])

    def _build_groups(self):
        groups = defaultdict(list)
        for img_path, class_number in self._images.items():
            group_label = self._class_mapping[class_number]
            groups[group_label].append(img_path)
        print(f"[FLICKR] {len(groups)} classes with a total of {sum([len(g) for g in groups.values()])} samples")
        return groups

    def read(self):
        folders = [f for f in listdir(self._path) if isdir(join(self._path, f))]
        for i, folder in enumerate(folders):
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

    def get_random_from_class(self, class_label: str) -> Tuple[str, int]:
        class_group = self._class_groups[class_label]
        label = self._class_mapping_inverted[class_label]
        sample = random.sample(class_group, 1)[0]
        return sample, label

    def get_random_from_non_class(self, class_label: str) -> Tuple[str, int]:
        negative_groups = [gk for gk in self._class_groups.keys() if gk != class_label]
        negative_class = random.sample(negative_groups, 1)[0]
        return self.get_random_from_class(negative_class)

    @property
    def class_mapping(self):
        return self._class_mapping

    @property
    def images(self):
        return self._images

    @property
    def class_mapping_inverted(self):
        return self._class_mapping_inverted

    @class_mapping.setter
    def class_mapping(self, value):
        self._class_mapping = value

    @class_mapping_inverted.setter
    def class_mapping_inverted(self, value):
        self._class_mapping_inverted = value


class ContrastiveDataset(Dataset):
    def __init__(self, flickr_dataset: FlickrDataset, sketches_dataset: SketchDataset, n_similar: int, m_different: int):
        super(ContrastiveDataset, self).__init__()
        self._n_similar = n_similar
        self._m_different = m_different
        self._flickr = flickr_dataset
        self._sketches = sketches_dataset
        self._pairs = self._create_pairs()

    def _create_pairs(self):

        pairs = []  # (first, second, similarity (1 or 0))
        for first_img_path, class_number in self._flickr.images.items():
            flickr_class_label = self._flickr.class_mapping[class_number]

            # get N similar to first image
            for second_img_path in random.sample(self._sketches.class_groups[flickr_class_label], self._n_similar):
                second_img_label = self._sketches.class_mapping_inverted[flickr_class_label]
                pairs.append((first_img_path, second_img_path, class_number, second_img_label, 1))

            # different M different to first image
            different_groups = [g for g in self._sketches.class_groups.keys() if g != flickr_class_label]
            different_group = random.sample(different_groups, 1)[0]
            for second_img_path in random.sample(self._sketches.class_groups[different_group], self._m_different):
                second_img_label = self._sketches.class_mapping_inverted[flickr_class_label]
                pairs.append((first_img_path, second_img_path, class_number, second_img_label, 0))

        print(f"Contrastive dataset has been created with a total of {len(pairs)} pairs")
        return pairs

    def __getitem__(self, item):
        # we took an image from flickr dataset and then we uniformly sample an sketch from sketches dataset
        # (flickr, sketch, similarity)
        img_flickr_path, img_sketches_path, img_flickr_label, img_sketches_label, target = self._pairs[item]
        img_flickr, img_sketches = Image.open(img_flickr_path), Image.open(img_sketches_path)

        img_flickr = self._flickr.process_image_pipeline(img_flickr)
        img_sketches = self._sketches.process_image_pipeline(img_sketches)
        return img_flickr, img_sketches, img_flickr_label, img_sketches_label, target

    def __len__(self):
        return len(self._pairs)

    def on_epoch_end(self):
        print("\nRefreshing dataset")
        self._pairs = self._create_pairs()


class TripletDataset(Dataset):
    def __init__(self, flickr_dataset: FlickrDataset, sketches_dataset: SketchDataset):
        super(TripletDataset, self).__init__()
        self._flickr = flickr_dataset
        self._sketches = sketches_dataset
        self._triplets, self._triplets_labels = self._create_triplets()

    def _create_triplets(self):
        triplets = []
        triplets_labels = []
        for group_label, group_members in self._sketches.class_groups.items():
            # get an anchor
            anchor_label = self._sketches.class_mapping_inverted[group_label]
            for group_member in group_members:
                anchor_path = group_member
                positive_path, positive_label = self._flickr.get_random_from_class(group_label)
                negative_path, negative_label = self._flickr.get_random_from_non_class(group_label)

                triplets.append((anchor_path, positive_path, negative_path))
                triplets_labels.append((anchor_label, positive_label, negative_label))
        print(f"{len(triplets)} triplets have been created")
        return triplets, triplets_labels

    def __getitem__(self, item):
        anchor_sketch_path, positive_flickr_path, negative_flickr_path = self._triplets[item]
        anchor_label, positive_label, negative_label = self._triplets_labels[item]

        # read images
        anchor = Image.open(anchor_sketch_path)
        positive = Image.open(positive_flickr_path)
        negative = Image.open(negative_flickr_path)

        # process them
        anchor = self._sketches.process_image_pipeline(anchor)
        positive = self._flickr.process_image_pipeline(positive)
        negative = self._flickr.process_image_pipeline(negative)

        return anchor, positive, negative, anchor_label, positive_label, negative_label

    def __len__(self):
        return len(self._triplets)

    def on_epoch_end(self):
        print("\nRefreshing dataset")
        self._triplets, self._triplets_labels = self._create_triplets()


class SimpleDataset(Dataset):
    def __init__(self, path: str):
        self._path = path
        self._images = self._read_images()
        self._all_images = []
        self._all_images_class = []
        for k, v in self._images.items():
            self._all_images.extend(v)
            self._all_images_class.extend([k] * len(v))

        self.process_image_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            RotationTransform(90),
            transforms.CenterCrop(224)
            # transforms.Normalize(self.mean, self.std),
        ])

    def _read_images(self):
        raise NotImplementedError

    def __getitem__(self, item):

        path = self._all_images[item]
        arr = Image.open(self._all_images[item])
        arr = self.process_image_pipeline(arr)
        label = self._all_images_class[item]
        return path, arr, label

    def __len__(self):
        return len(self._all_images)

    def get_process_pipeline(self):
        return self.process_image_pipeline


class ImageFlickr15K(SimpleDataset):

    def __init__(self, path: str):
        super(ImageFlickr15K, self).__init__(path)

    def _read_images(self):
        """
        Read the paths of images and organize them by class.
        """
        images = {}
        folders = glob.glob(os.path.join(self._path, "images/*"))
        for folder in folders:
            class_label = int(folder.split("/")[-1])
            images_path = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
            images[class_label] = images_path
        return images


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_sketches_dataset = SketchTrainDataset('/home/rudy/Documents/cc7221/tarea2/data/Sketch_EITZ')
    train_loader = DataLoader(train_sketches_dataset, batch_size=8)
    for images, labels in train_loader:
        print(images.shape)
        break

    train_flickr_dataset = FlickrDataset('/home/rudy/Documents/cc7221/tarea2/data/Flickr25K',
                                         class_mapping=train_sketches_dataset.class_mapping)
    train_flickr_loader = DataLoader(train_flickr_dataset, batch_size=8)
    for images, labels in train_flickr_loader:
        print(images.shape)
        break

    print("\nCreating contrastive dataset")
    train_contrastive = ContrastiveDataset(train_flickr_dataset, train_sketches_dataset, 1, 1)
    d = train_contrastive[0]
    train_contrastive_loader = DataLoader(train_contrastive, batch_size=8)
    for flickr_images, _, _, _, _ in train_contrastive_loader:
        print(flickr_images.shape)
        break

    print("\nCreating triplet dataset ...")
    train_triplet = TripletDataset(train_flickr_dataset, train_sketches_dataset)
    d = train_triplet[0]
    train_triplet_loader = DataLoader(train_triplet, batch_size=8)
    for anchors, _, _, _, _, _ in train_triplet_loader:
        print(anchors.shape)
        break
