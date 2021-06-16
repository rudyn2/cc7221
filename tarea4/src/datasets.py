from torch.utils.data import Dataset
import glob
from pathlib import Path
import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import random


class CustomTransform:
    """Rotate, horizontal flip and vertical flip."""

    def __init__(self, p_flip: float = 0.5, mode: str = "train"):
        self.angles = list(np.linspace(-90, 90, num=37))
        self.p_flip = p_flip
        self.mode = mode
        self._device = "cuda"

    def __call__(self, image, mask):
        image = torch.tensor(image, device=self._device) / 255.0
        image = image.repeat(3, 1, 1)
        mask = torch.tensor(mask, device=self._device)
        mask = mask / 255.0 if len(mask.shape) > 2 else mask

        if self.mode == "train":
            angle = random.choice(self.angles)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            # vertical flip
            if random.random() < self.p_flip:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # horizontal flip
            if random.random() < self.p_flip:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        return image, mask


class SpermDataset(Dataset):

    def __init__(self, path: str, target_one_hot: bool = False):
        """
        SpermDataset Constructor.
        :param path: Path to dataset folder.
        :param target_one_hot: If true, the target will receive one hot encoding formatting.
                               Output dimension will be [CHANNELS, NUM_CLASSES, WIDTH, HEIGHT]
                               instead of [CHANNELS, WIDTH, HEIGHT].
        """
        self._device = "cuda"
        self.path = path
        self.mode = "train"
        self.data = {"train": {}, "test": {}}
        self.use_one_hot = target_one_hot
        self._load()
        self.segmentation_masks = self._build_segmentation()
        self.idx_to_key = {
            "train": list(self.data["train"].keys()),
            "test": list(self.data["test"].keys())
        }

        self.transform = {
            "train": CustomTransform(mode="train"),
            "test": CustomTransform(mode="test")

        }

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "test"

    def _load(self):
        # load images
        folders = os.listdir(str(Path(self.path).joinpath("images")))
        for folder in folders:
            files = glob.glob(str(Path(self.path).joinpath("images").joinpath(folder).joinpath("*.jpg")))
            images = {Path(file_path).name: cv2.imread(file_path, 0) for file_path in files}
            self.data[folder] = images

    def _build_segmentation(self):
        # build segmentation masks
        seg = {}
        for idx in range(1, 21):
            file_name = f"Placa1-imagen{idx}.jpg"
            mask = []
            for mask_folder in ["Head-Masks", "Midpiece-Masks", "Tail-Masks"]:
                file_path = Path(self.path).joinpath("mask").joinpath(mask_folder).joinpath(file_name)
                mask_image = cv2.imread(str(file_path), 0)
                mask.append(mask_image)

            # little dirty code to avoid missing masks
            if mask[0] is None:
                continue

            file_mask = np.stack(mask, axis=0)
            if not self.use_one_hot:
                file_mask = np.argmax(file_mask, axis=0)
            seg[file_name] = file_mask
        return seg

    def __getitem__(self, item: int):
        # the file name is used for indexation
        file_name = self.idx_to_key[self.mode][item]

        image = self.data[self.mode][file_name]
        mask = self.segmentation_masks[file_name]

        image_transformed, mask_transformed = self.transform[self.mode](image, mask)
        return image_transformed, mask_transformed

    def __len__(self):
        return len(self.data[self.mode])


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    folder_path = r"C:\Users\C0101\PycharmProjects\cc7221\tarea4\data\SpermSegGS"
    dataset = SpermDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=2)
    for batch in dataloader:
        break
