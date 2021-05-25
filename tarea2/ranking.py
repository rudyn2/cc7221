from typing import List, Tuple
import os
import torch.nn
from models import SiameseNetwork
import re
from PIL import Image
from torchvision.transforms import transforms
from tarea2.utils import RotationTransform
from tqdm import tqdm
from torch.utils.data import DataLoader
from feats import ImageFlickrFeatures


class Ranker(object):

    def __init__(self,
                 path: str,
                 image_dataset_features: ImageFlickrFeatures,
                 feature_extractor: SiameseNetwork,
                 similarity_fn: torch.nn.Module):

        self._device = 'cpu'
        self._feature_extractor = feature_extractor
        self._feature_extractor.to(self._device)
        self._feature_extractor.eval()

        self._similarity_fn = similarity_fn
        self._path = path
        self._image_dataset_features = image_dataset_features
        self._process_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            RotationTransform(90)
        ])
        self._query_class = self._read_query_class()
        self._batch_size = 8

    def _read_query_class(self):
        query_class_mapping = {}
        with open(os.path.join(self._path, "query_class.txt"), "r") as f:
            lines = f.readlines()

        for line in lines:
            s = re.split(r'\t+', line.rstrip().rstrip('\t'))
            full_path = os.path.join(self._path, "queries", s[0])
            query_class_mapping[full_path] = int(s[1])
        return query_class_mapping

    def get_rank(self, q: str) -> Tuple[int, List[Tuple[str, int, float]]]:
        """
        Returns a sorted list of tuples that contains the path to contrastive image and its similarity

        Example:
            .get_rank('flickr15k/queries/0001.png') -> 2, [('flickr15k/images/0024.png', 10, 0.98),
                                                           ('flickr15k/images/0022.png', 23, 0.97), ...]
        """
        q = os.path.realpath(q)
        assert q in self._query_class.keys()

        # load query image
        query_class = self._query_class[q]
        query_img_tensor = self._process_pipeline(Image.open(q))
        query_img_tensor = query_img_tensor.to(self._device)
        loader = DataLoader(self._image_dataset_features, batch_size=self._batch_size)

        # extract its features and create a mini-batch
        pred = self._feature_extractor.extract_features_sketch(query_img_tensor.unsqueeze(0))
        batch_query = torch.cat([pred]*self._batch_size)

        similarities = []
        for paths, features, labels in tqdm(loader, "Querying"):

            # calculate distances
            distances = self._similarity_fn(features, batch_query[:len(labels), :])
            distances = list(distances.detach().cpu().numpy())
            labels = list(labels.detach().cpu().numpy())

            # group similarities
            group_similarities = list(zip(list(paths), labels, distances))
            similarities.extend(group_similarities)

        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)

        return query_class, similarities


if __name__ == '__main__':

    from models import ResNet34
    from similarities import CosineSimilarity
    from feats import ImageFlickrFeatures

    imagenet_net = ResNet34()
    sketches_net = ResNet34()

    siamese_net = SiameseNetwork(sketches_net, imagenet_net)
    siamese_net.load_state_dict(torch.load("weights/best_SiameseNetwork_contrastive.pth"))

    s = torch.nn.CosineSimilarity()
    flickr_dataset = ImageFlickrFeatures("dbs/features.db")
    r = Ranker('/home/rudy/Documents/cc7221/tarea2/data/Flickr15K',
               image_dataset_features=flickr_dataset,
               feature_extractor=siamese_net,
               similarity_fn=s)
    rank = r.get_rank('data/Flickr15K/queries/1001.png')
