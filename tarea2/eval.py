import argparse
from collections import defaultdict
from typing import Tuple, List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
from os import listdir
# from dataset import SketchTestDataset
from models import ResNet34
from metrics import *
from ranking import Ranker
from feats import ImageFlickrFeatures
from models import SiameseNetwork
from distances import *


class Evaluator:

    def __init__(self, path_wieght: str, path_data: str, similarity):
        self.path_weight = path_wieght
        self.path_data = path_data
        self.similarity = similarity
        self.flickr_dataset = ImageFlickrFeatures("dbs/features.db")
        # self.ranking = ranking

        imagenet_net = ResNet34()
        sketches_net = ResNet34()

        # print("Adapting output layers...")
        sketches_net.adapt_fc()
        imagenet_net.adapt_fc()

        siamese_net = SiameseNetwork(sketches_net, imagenet_net)
        siamese_net.load_state_dict(torch.load(
            self.path_weight))  # r'C:\Users\aleja\Desktop\Tareas\Reconocimiento Virtual con Deep Learning\T2\best_SiameseNetwork_contrastive.pth'
        self.net = siamese_net
        self.ranking = Ranker(self.path_data,
                              image_dataset_features=self.flickr_dataset,
                              feature_extractor=self.net,
                              similarity_fn=self.similarity)




    def calc_rank(self, path_img):
        rank = self.ranking.get_rank(path_img)
        return rank

    def calc_all_ranks(self, path_querys):
        self.imgs_names = listdir(path_querys)
        for i in range(len(self.imgs_names)):
            self.imgs_names[i] = path_querys + '/' + self.imgs_names[i]

        self.classes = []
        self.ranks = []
        for i in range(len(self.imgs_names)):
            c, rank = self.ranking.get_rank(self.imgs_names[i])
            self.classes.append(c)
            self.ranks.append(rank)

    def calc_map(self):
        mean_ap = map(self.classes, self.ranks)
        return mean_ap

    def calc_recall_ratio(self, len_class_path: str):
        x, y = recall_ratio_tot(self.classes, self.ranks, len_class_path)
        plt.plot(x, y)
        plt.xlabel('Recall')
        plt.ylabel('Retrieved images')
        plt.title('Recall ratio Curve')
        return x, y

    def calc_recall_prec(self, len_class_path: str):
        rp = recall_prec_tot(self.classes, self.ranks, len_class_path)
        rec = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.plot(rec, rp)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Recall-Precision Curve')
        return rp


# in contruction


if __name__ == '__main__':
    pw = r'C:\Users\aleja\Desktop\Tareas\Reconocimiento Virtual con Deep Learning\T2\best_SiameseNetwork_contrastive.pth'
    pd = 'B:\Flickr\Flickr15K'
    l_c = 'B:\Flickr\Flickr15K\images'
    sim = torch.nn.CosineSimilarity()
    a = Evaluator(pw, pd, sim)
    pq = r'B:\Flickr\Flickr15K\queries'
    a.calc_all_ranks(pq)
    mapita = a.calc_map()
    x, y = a.calc_recall_ratio(l_c)
    rp = a.calc_recall_prec(l_c)
    rp = a.calc_recall_prec(l_c)