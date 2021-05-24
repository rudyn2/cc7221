import torch
import numpy as np
from ranking import Ranker
from datasets import FlickrDataset2


def simple_accuracy(y_true, y_pred):
    correct_prediction = torch.eq(torch.argmax(y_true, 1), torch.argmax(y_pred, 1))

    cont = 0
    for i in range(len(correct_prediction)):
        if correct_prediction[i].item():
            cont += 1
    acc = cont / len(y_pred)

    return acc


def metric_accuracy_siamese(y_true_a, y_true_p, y_true_n, cl_a, cl_p, cl_n):
    # y_true_a, y_true_p, y_true_n = torch.split(y_true, split_size_or_sections=3, dim=1)
    correct_prediction_a = torch.eq(torch.argmax(torch.squeeze(y_true_a), 1), torch.argmax(torch.squeeze(cl_a), 1))
    correct_prediction_p = torch.eq(torch.argmax(torch.squeeze(y_true_p), 1), torch.argmax(torch.squeeze(cl_p), 1))
    correct_prediction_n = torch.eq(torch.argmax(torch.squeeze(y_true_n), 1), torch.argmax(torch.squeeze(cl_n), 1))

    cont_a = 0
    cont_p = 0
    cont_n = 0

    for i in range(len(correct_prediction_a)):
        if correct_prediction_a[i].item():
            cont_a += 1
    for i in range(len(correct_prediction_p)):
        if correct_prediction_p[i].item():
            cont_p += 1
    for i in range(len(correct_prediction_n)):
        if correct_prediction_n[i].item():
            cont_n += 1

    acc_a = cont_a / len(cl_a)
    acc_p = cont_p / len(cl_p)
    acc_n = cont_n / len(cl_n)
    acc = (acc_a + acc_p + acc_n) / 3.0
    return acc




def average_precision(c, rank):  # o dataset
    # caso1 ranking contiene la clase de la imagen al igual q true
    #c, rank = ranking.get_rank(path)
    cont = 0
    cont2 = 0
    for j in range(len(rank)):
        if c == rank[j][1]:
            cont += (1 + cont2) / (j + 1)
            cont2 += 1

    ap = cont / cont2
    #print(cont2)

    # map = contf / len(y_true)
    return ap




def map(c:list, rank:list):
    aps = []
    for i in range(len(c)):
        aps.append(average_precision(c[i], rank[i]))
    return np.mean(aps)



def recall_ratio_per_query(c, rank, len_class):
    cont = 0
    x = []
    y = []
    for i in range(len(rank)):  # or 2000
        x.append(i + 1)
        if rank[i][1] == c:
            cont += 1
        y.append(cont / len_class)
    return x, y

def recall_ratio_tot(c:list, rank:list, path_db: str ):#'B:\Flickr\Flickr15K\images'
    test_flickr_db = FlickrDataset2(path_db)
    img_per_class = test_flickr_db._build_groups()
    len_classes = []
    for i in range(1, 34):
        len_classes.append(len(img_per_class[str(i)]))

    rrx = []
    rry = []
    for i in range(len(c)):
        rr_qx, rr_qy = recall_ratio_per_query(c[i], rank[i], len_classes[c[i] - 1])
        rrx.append(rr_qx)
        rry.append(rr_qy)
    return np.mean(rrx,0), np.mean(rry,0)





def recall_presicion_per_query(c,rank, len_class):
    recalls = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #c, rank = ranking.get_rank(path)
    precision = np.zeros(11)
    cont = 0
    cont2 = 0

    for i in range(len(rank)):
        if rank[i][1] == c:
            cont = (1 + cont2) / (i + 1)
            cont2 += 1
            idx = int((cont2 / len_class) * 10)
            for j in range(idx+1):
                if cont > precision[j]:
                    precision[j] = cont
    return precision

def recall_prec_tot(c:list, rank:list, path_db: str ):
    test_flickr_db = FlickrDataset2(path_db)
    img_per_class = test_flickr_db._build_groups()
    len_classes = []
    for i in range(1,34):
        len_classes.append(len(img_per_class[str(i)]))

    rps = []
    for i in range(len(c)):
        rp = recall_presicion_per_query(c[i],rank[i],len_classes[c[i]-1])
        rps.append(rp)
    return np.mean(rps,0)












