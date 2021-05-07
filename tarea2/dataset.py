import os
from abc import ABC, abstractmethod
import re
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np


class SketchDataset(ABC, Dataset):

    # MEAN = [0.9818744]
    # STD = [0.0077032577]

    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]

    def __init__(self, path: str):
        self._path = path
        self._metadata_path = os.path.join(self._path, self.get_txt_name())
        self._images = self._read_metadata()            # image_path -> class
        self._images_paths = list(self._images.keys())

        self.process_image_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            # transforms.Normalize(self.mean, self.std),
        ])

    @abstractmethod
    def get_txt_name(self):
        raise NotImplementedError

    def _read_metadata(self):
        with open(self._metadata_path, "r") as f:
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


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_dataset = SketchTrainDataset('/home/rudy/Documents/cc7221/tarea2/data/Sketch_EITZ')
    train_loader = DataLoader(train_dataset, batch_size=8)
    for i in range(100):
        img = train_dataset[i]
    for images, labels in train_loader:
        print(images.shape)

