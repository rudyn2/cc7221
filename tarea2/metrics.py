import torch


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


def mean_average_precision(y_pred_ranks, y_true):  # o dataset
    # caso1

    # caso2
    contf = 0
    for i in range(len(y_true)):
        cont = 0
        for j in range(len(y_pred_ranks[i])):
            if y_true[i] == y_pred_ranks[i][j]:
                cont += 1
        contf += cont / len(y_pred_ranks[i])

    map = contf / len(y_true)
    return map


def recall(ranks, y_true):
    contf = 0
    for i in range(len(y_true)):
        cont = 0
        for j in range(len(ranks[i])):
            if y_true[i] == ranks[i][j]:
                cont += 1
        contf += cont / len(ranks[i])

    recall = contf / len(y_true)
    return recall


def Precision(ranks, y_true):
    contf = 0
    for i in range(len(y_true)):
        cont = 0
        for j in range(len(ranks[i])):
            if y_true[i] == ranks[i][j]:
                cont += 1
        contf += cont / len(ranks[i])

    prec = contf / len(ranks)
    return prec

# def recall(y_true,y_pred) #dudas hacerlo con keras y entradas
