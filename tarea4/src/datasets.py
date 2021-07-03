from torch.utils.data import Dataset
import glob
from pathlib import Path
import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
from typing import Tuple


def mixup(image_i: np.ndarray,
          image_j: np.ndarray,
          mask_i: np.ndarray,
          mask_j: np.ndarray,
          alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    lam = np.random.beta(alpha, alpha)
    mixup_image = image_i * lam + image_j * (1 - lam)
    mixup_mask = mask_i * lam + mask_j * alpha
    return mixup_image, mixup_mask


def random_crop_mask(arr: np.ndarray, min_area: int = 9000, max_area: int = 50000):
    """
    Creates a random boolean mask with a minimal area.
    """
    found = False
    first_point, second_point = (0, 0), arr.shape
    points = []
    for _ in range(10):
        fp = int(arr.shape[0] * random.random() * 0.8), int(arr.shape[1] * random.random() * 0.8)
        sp_x = int(fp[0] + (arr.shape[0] - fp[0]) * random.random())
        sp_y = int(fp[1] + (arr.shape[1] - fp[1]) * random.random())
        area = (fp[0] - sp_x) * (fp[1] - sp_y)
        if area > min_area:
            if area < max_area:
                first_point = fp
                second_point = (sp_x, sp_y)
                found = True
                break
            points.append((fp, (sp_x, sp_y), area))
            continue
        points.append((fp, (sp_x, sp_y), area))

    if not found:
        points.sort(reverse=True, key=lambda x: x[2])
        first_point = points[0][0]
        second_point = points[0][1]
    mask = np.zeros_like(arr)
    mask[first_point[0]:second_point[0], first_point[1]:second_point[1]] = 1
    return mask > 0


class CustomTransform:
    """Rotate, horizontal flip and vertical flip."""

    def __init__(self, p_flip: float = 0.5, p_crop: float = 0.5, mode: str = "train", new_size: tuple = None):
        self.angles = list(np.linspace(-90, 90, num=37))
        self.p_flip = p_flip
        self.mode = mode
        self.new_size = new_size
        self._device = "cuda"
        self.p_crop = p_crop

    def to_tensor(self, arr: np.array, normalize: bool = False, repeat_channels: int = None) -> torch.Tensor:
        """
        Transform an array to a torch Tensor of type Float64 and shape [C, W, H]
        """
        if normalize:
            arr = arr / arr.max()
        else:
            arr = arr * 1.0
        arr = torch.tensor(arr, device=self._device)
        if len(arr.shape) <= 2:
            arr = arr.unsqueeze(0)
        if repeat_channels:
            arr = arr.repeat(repeat_channels, 1, 1)
        return arr

    def __call__(self, image, mask):
        image = self.to_tensor(image, normalize=True, repeat_channels=3).float()
        mask = self.to_tensor(mask, normalize=False).type(torch.LongTensor).to(self._device)

        if self.new_size:
            image = TF.resize(image, size=list(self.new_size))
            mask = TF.resize(mask, size=list(self.new_size), interpolation=TF.InterpolationMode.NEAREST)

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

            if random.random() < self.p_crop:
                # Resize
                resize = T.Resize(size=(480, 672), interpolation=TF.InterpolationMode.NEAREST)
                # Random crop
                i, j, h, w = T.RandomCrop.get_params(
                    image, output_size=(320, 448))
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)

                image = resize(image)
                mask = resize(mask)

            # Contrast
        return image, mask


def get_datasets(path: str, use_validation: bool = False, **kwargs):

    # load images
    data = {"train": {}, "test": {}}
    folders = os.listdir(str(Path(path).joinpath("images")))
    for folder in folders:
        files = glob.glob(str(Path(path).joinpath("images").joinpath(folder).joinpath("*.jpg")))
        images = {Path(file_path).name: cv2.imread(file_path, 0) for file_path in files}
        data[folder] = images

    # random split train and validation sets
    train_val_keys = list(data["train"].keys())
    if use_validation:
        val_keys = ['Placa1-imagen13.jpg', 'Placa1-imagen18.jpg', 'Placa1-imagen1.jpg']
    else:
        val_keys = []
    train_keys = [k for k in train_val_keys if k not in val_keys]

    # create datasets and return them
    train_dataset = SpermDataset(path, data["train"], train_keys, transform_mode="train", **kwargs)
    val_dataset = SpermDataset(path, data["train"], val_keys, transform_mode="val", **kwargs)
    test_dataset = SpermDataset(path, data["test"], list(data["test"].keys()), transform_mode="val", **kwargs)
    return train_dataset, val_dataset, test_dataset


class SpermDataset(Dataset):

    def __init__(self, path: str,
                 data: dict, keys: list,
                 target_one_hot: bool = False,
                 transform_mode: str = "train",
                 mosaic_prob: float = 0.5,
                 new_size: tuple = None):
        """
        SpermDataset Constructor.
        :param path: Path to dataset folder.
        :param target_one_hot: If true, the target will receive one hot encoding formatting.
                               Output dimension will be [CHANNELS, NUM_CLASSES, WIDTH, HEIGHT]
                               instead of [CHANNELS, WIDTH, HEIGHT].
        :param mosaic_prob: probability of applying mosaic transform
        """
        self._device = "cuda"
        self.path = path
        self.data = data
        self.idx_to_key = keys
        self._mosaic_prob = mosaic_prob

        self.use_one_hot = target_one_hot
        self.segmentation_masks = self._build_segmentation()
        self.transform_mode = transform_mode
        self.transform = {
            "train": CustomTransform(mode="train", new_size=new_size),
            "val": CustomTransform(mode="test", new_size=new_size),
            "test": CustomTransform(mode="test", new_size=new_size)
        }

    def _build_segmentation(self):
        # build segmentation masks
        seg = {}
        for idx in range(1, 21):
            file_name = f"Placa1-imagen{idx}.jpg"
            mask = []
            for mask_folder in ["Head-Masks", "Midpiece-Masks", "Tail-Masks"]:
                file_path = Path(self.path).joinpath("mask").joinpath(mask_folder).joinpath(file_name)
                mask_image = cv2.imread(str(file_path), 0)
                if mask_image is not None:
                    # mask_image = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
                    mask_image = (mask_image > 200).astype(np.uint8) * 255
                mask.append(mask_image)

            # little dirty code to avoid missing masks
            if mask[0] is None:
                continue

            mask.insert(0, (~((mask[0] + mask[1] + mask[2]) > 0)) * 255)  # add background mask
            file_mask = np.stack(mask, axis=0)

            if not self.use_one_hot:
                file_mask = np.argmax(file_mask, axis=0)

            seg[file_name] = file_mask
        return seg

    def __getitem__(self, item: int):
        # the file name is used for indexation
        file_name = self.idx_to_key[item]

        image = self.data[file_name]
        mask = self.segmentation_masks[file_name]

        # apply mosaic transform
        if self.transform_mode == 'train' and random.random() < self._mosaic_prob:
            second_file_name = random.choice(list(set(self.idx_to_key) - set(file_name)))
            second_image = self.data[second_file_name]
            second_mask = self.segmentation_masks[second_file_name]
            random_mask = random_crop_mask(image)
            random_mask = np.logical_and(random_mask, second_mask != 0)     # we just select the sperms
            image[random_mask] = second_image[random_mask]
            mask[random_mask] = second_mask[random_mask]

        image_transformed, mask_transformed = self.transform[self.transform_mode](image, mask)
        return image_transformed, mask_transformed

    def __len__(self):
        return len(self.idx_to_key)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    folder_path = r"C:\Users\C0101\PycharmProjects\cc7221\tarea4\data\SpermSegGS"
    train, val, test = get_datasets(folder_path, use_validation=True, new_size=(480, 672))
    dataloader = DataLoader(train, batch_size=2)
    for batch in dataloader:
        break
