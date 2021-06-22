import argparse
import pickle
from datasets import ImageFlickr15K
from models import SiameseNetwork
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np


class ImageFlickrFeatures(Dataset):

    def __init__(self, path: str):
        self._path = path
        self._data = None
        self._image_paths = None
        self.load()

    def load(self):
        with open(self._path, 'rb') as handle:
            self._data = pickle.load(handle)
            self._image_paths = list(self._data.keys())
        print(f"{len(self._data)} images loaded successfully")

    def get_paths(self):
        return list(self._data.keys())

    def __getitem__(self, item):
        path = self._image_paths[item]
        feat, label = self._data[path]
        feat = np.squeeze(feat)
        return path, feat, label

    def __len__(self):
        return len(self._data)


class ImageFlickrSaver(object):

    def __init__(self,
                 output_path: str,
                 model: SiameseNetwork,
                 image_dataset: ImageFlickr15K):

        self._output_path = output_path
        self._model = model
        self._image_dataset = image_dataset
        self._batch_size = 8

    def export(self):
        loader = DataLoader(self._image_dataset, batch_size=self._batch_size)
        data = {}

        for path, images, labels in tqdm(loader, "Exporting"):
            features = self._model.extract_features_image(images)
            features = features.detach().cpu().numpy()
            features = np.vsplit(features, images.shape[0])
            labels = list(labels.detach().cpu().numpy())
            batch_feats = dict(zip(path, list(zip(features, labels))))
            data.update(batch_feats)

        with open(self._output_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Features exported successfully at: {self._output_path}")


if __name__ == '__main__':
    from models import ResNet34
    import torch

    parser = argparse.ArgumentParser(description="Train model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--siamese-weights', default=r'C:\Users\aleja\Desktop\Tareas\Reconocimiento Virtual con Deep Learning\T2\best_SiameseNetwork_contrastive.pth',#'weights/best_SiameseNetwork_triplet.pth',
                        type=str, help='Path to Siamese network weights')
    parser.add_argument('--flickr-15k', default=r'B:/Flickr/Flickr15K', type=str,
                        help='Path to flickr dataset folder')
    parser.add_argument('--output', default='features_contrastive.db', type=str, help='Output path of feature db')

    args = parser.parse_args()

    imagenet_net = ResNet34()
    sketches_net = ResNet34()

    print("[*] Adapting output layers and loading weights...")
    siamese_net = SiameseNetwork(sketches_net, imagenet_net)
    siamese_net.load_state_dict(torch.load(args.siamese_weights))
    print("[+] Done!")

    dataset = ImageFlickr15K(path=args.flickr_15k)
    saver = ImageFlickrSaver(args.output, model=siamese_net, image_dataset=dataset)
    saver.export()

