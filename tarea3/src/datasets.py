import glob
import os
from pathlib import Path
from typing import Tuple, Dict, List

import albumentations as A
import cv2
import torch
import torchvision.ops
from torch.tensor import Tensor
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import re


class OrandCarDataset(Dataset):
    """
    Auxiliary class created to handle orand-car-with-bbs training data.
    """

    def __init__(self, path: str):
        super(OrandCarDataset, self).__init__()
        self._images = []
        self._path = path

        self._read()
        self.transform = A.Compose([
            A.Resize(65, 271),  # here we can add other transformations as well
            A.CLAHE(),
            A.GaussianBlur(),
            A.GaussNoise(),
            A.RandomBrightnessContrast(),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.1, label_fields=['labels']))

        # self._images = self._images[:100]

    def _read(self):
        training_folder = os.path.join(self._path, "training")
        if os.path.exists(training_folder):
            # read images
            images = glob.glob(os.path.join(training_folder, "images/*"))
            self._images = [Path(img) for img in images]

            # read annotations
            self._annotations = {}
            annotations = glob.glob(os.path.join(training_folder, "annotations/*"))
            for a in annotations:
                name, bboxes = self._process_annotation(a)
                self._annotations[name] = bboxes

            # filter, remove every image that doesn't has an annotation
            self._images = [img_path for img_path in self._images if img_path.stem in self._annotations.keys()]

            # verify that every image has an annotation
            assert all([img.stem in self._annotations.keys() for img in self._images])
            print(f"{len(self._images)} images were loaded successfully")
        else:
            raise ValueError("We couldn't find the training folder in the specified path.")

    @staticmethod
    def _process_annotation(path: str) -> Tuple[str, List[dict]]:
        """
        Read an annotation file and return the name of the associated image and a list of dictionaries, one per each bbox
        """
        with open(path, "r") as f:
            lines = f.readlines()
        name = Path(path).stem
        bboxes = []
        for line in lines:
            s = line.rstrip().split(":")
            class_label = int(s[0])
            bbox_coords = [max(0, int(c)) for c in s[1].split(",")]
            bboxes.append({
                "box": bbox_coords,
                "label": class_label,
            })
        return name, bboxes

    def __getitem__(self, item) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Return an image (C, H, W) in range [0, 1] and a dictionary with its bounding boxes in the following format:

        boxes (FloatTensor[N, 4]):
            the ground-truth boxes in [x1, y1, x2, y2] format,
            with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
        labels (Int64Tensor[N]):
            the class label for each ground-truth box
        """
        image_path = self._images[item]
        bboxes = self._annotations[image_path.stem]

        # parse bboxes metadata

        bboxes_list = [b['box'] for b in bboxes]
        labels_list = [b['label'] for b in bboxes]
        image = cv2.imread(str(image_path))

        transformed = self.transform(image=image, bboxes=bboxes_list, labels=labels_list)
        transformed_image = transformed['image'] / 255.0

        # parse bboxes from xywh to xyxy
        transformed_bboxes = torch.stack([torch.tensor(b) for b in transformed['bboxes']])
        transformed_bboxes = torchvision.ops.box_convert(transformed_bboxes, in_fmt='xywh', out_fmt='xyxy')

        # parse labels
        transformed_labels = torch.stack([torch.tensor(l) for l in transformed['labels']])
        return transformed_image, dict(boxes=transformed_bboxes, labels=transformed_labels)

    def __len__(self) -> int:
        return len(self._images)


class OrandCarTestDataset(Dataset):

    def __init__(self, path: str):
        super(OrandCarTestDataset, self).__init__()
        self._images = []
        self._labels = {}
        self._path = path

        self._read()
        self.transform = A.Compose([
            A.Resize(65, 271),  # here we can add other transformations as well
            ToTensorV2(),
        ])

    def _read(self):
        test_folder = os.path.join(self._path, "test")
        if os.path.exists(test_folder):
            # read images
            images = glob.glob(os.path.join(test_folder, "images/*"))
            self._images = [Path(img) for img in images]

            # read labels
            with open(os.path.join(test_folder, "list.txt"), "r") as f:
                lines = f.readlines()
            for line in lines:
                s = re.split(r'\t+', line.rstrip().rstrip('\t'))
                self._labels[Path(s[0]).stem] = s[1]

            # filter, remove every image that doesn't has an annotation
            self._images = [img_path for img_path in self._images if img_path.stem in self._labels.keys()]

            # verify that every image has an annotation
            assert all([img.stem in self._labels.keys() for img in self._images])
            print(f"{len(self._images)} images were loaded successfully")
        else:
            raise ValueError("We couldn't find the training folder in the specified path.")

    def __getitem__(self, item: int) -> Tuple[Tensor, str]:
        """
         Return an image (C, H, W) in range [0, 1] and its label as an string
        """
        image_path = self._images[item]
        image = cv2.imread(str(image_path))
        transformed = self.transform(image=image)
        transformed_image = transformed['image'] / 255.0
        label = self._labels[image_path.stem]
        return transformed_image, label

    def __len__(self):
        return len(self._images)


def custom_collate_fn(batch):
    images_batch = torch.stack([t[0] for t in batch])
    bboxes_list = [t[1] for t in batch]
    return images_batch, bboxes_list


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    dataset_folder = "/home/rudy/Documents/cc7221/tarea3/data/orand-car-with-bbs"
    d = OrandCarDataset(dataset_folder)

    print("Visualizing random image")
    image, targets = d[0]
    image = (image * 255).type(torch.ByteTensor)
    labels_ = [str(s) for s in list(targets['labels'].numpy())]
    to_show = torchvision.utils.draw_bounding_boxes(image, targets['boxes'], labels_)

    plt.imshow(to_show.permute(1, 2, 0))
    plt.show()

    print("\nTesting test loader...")
    test_dataset = OrandCarTestDataset(dataset_folder)
    test_loader = DataLoader(test_dataset, batch_size=8)
    for images, labels in tqdm(test_loader, "Verifying"):
        continue

    print("\nTesting training loader...")
    train_loader = DataLoader(d, batch_size=8, collate_fn=custom_collate_fn)
    for images, bboxes in tqdm(train_loader, "Verifying"):
        continue


